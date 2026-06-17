"""Load a PRECOMPUTED EpBench book + Q&A directly, bypassing the generation
pipeline (BenchmarkGenerationWrapper.__end2end).

Why: __end2end always re-derives events/paragraphs/qa and asserts the result
matches the stored data. That path is fragile under newer libraries
(Python 3.14 / pandas 3.x): q_idx reordering breaks the self-consistency
assert and the paragraph loader index-errors. The figshare data is final, so
we just LOAD book.json + the parquets and expose the SAME accessors the
EvaluationWrapper + answer/judge generators use. No regeneration, no API,
no asserts.

Deliberately imports NOTHING from epbench.src.generation (that chain pulls
heavy/fragile deps); nb_tokens/nb_chapters are read from the canonical book
dir name, and split_chapters is replicated inline (same regex as
printing.split_chapters_func). Drop-in for BenchmarkGenerationWrapper on the
prompting / rag / compresh paths.
"""
from pathlib import Path
import ast
import glob
import json
import re
import pandas as pd

# nb_events -> nb_chapters in the precomputed default books (from the figshare data)
_NB_EVENTS_TO_CHAPTERS = {20: 19, 200: 196, 2000: 1967}


def _split_chapters(book):
    """Same as printing.split_chapters_func: {chapter_num: content}."""
    pattern = r'Chapter (\d+)\n\n(.*?)(?=Chapter \d+\n\n|$)'
    out = {}
    for chapter, content in re.findall(pattern, book, re.DOTALL):
        out[int(chapter)] = content.strip()
    return out


class LoadedBenchmark:
    def __init__(self, prompt_parameters, model_parameters, book_parameters, data_folder):
        self.prompt_parameters = prompt_parameters
        self.model_parameters = model_parameters
        self.book_parameters = book_parameters

        data_folder = Path(data_folder)
        nb_events = prompt_parameters['nb_events']
        nb_chapters = _NB_EVENTS_TO_CHAPTERS.get(nb_events, nb_events)
        seed = prompt_parameters.get('seed', 0)
        udir = data_folder / (
            f"U{prompt_parameters['name_universe']}_S{prompt_parameters['name_styles']}_seed{seed}")
        model = model_parameters['model_name']
        pattern = str(udir / "books" / f"model_{model}_*nbchapters_{nb_chapters}_*")
        matches = sorted(glob.glob(pattern))
        if not matches:
            raise FileNotFoundError(
                f"No precomputed book matching:\n  {pattern}\n"
                f"(nb_events={nb_events} -> nb_chapters={nb_chapters}). "
                f"Check the figshare data is unzipped under {udir}/books/.")
        self.book_dir = Path(matches[0])

        # nb_chapters / nb_tokens from the canonical dir name (these exact
        # numbers built the answer output paths, so parsing them matches).
        m = re.search(r'nbchapters_(\d+)_nbtokens_(\d+)', self.book_dir.name)
        self._nb_chapters = int(m.group(1)) if m else nb_chapters
        self._nb_tokens = int(m.group(2)) if m else 0

        with open(self.book_dir / "book.json") as f:
            self.book = json.load(f)  # single string; chapters separated by '\n\n\n'

        df_qa = pd.read_parquet(self.book_dir / "df_qa.parquet", engine="pyarrow")
        # post-load conversions (mirror BenchmarkGenerationWrapper.__end2end)
        df_qa['correct_answer_detailed'] = [
            ast.literal_eval(x) if isinstance(x, str) else x
            for x in df_qa['correct_answer_detailed']]
        if 'debug_changed' in df_qa.columns:
            df_qa['debug_changed'] = [set(x) for x in df_qa['debug_changed']]
        self.df_qa = df_qa

        dfg = pd.read_parquet(self.book_dir / "df_book_groundtruth.parquet", engine="pyarrow")
        if 'post_entities' in dfg.columns:
            dfg['post_entities'] = [set(x) for x in dfg['post_entities']]
        self.df_book_groundtruth = dfg

        self.split_chapters = _split_chapters(self.book)
        print(f"[LoadedBenchmark] {self.book_dir.name} · {len(self.df_qa)} questions · "
              f"{self._nb_chapters} chapters · {self._nb_tokens} tokens")

    # ── accessors expected by EvaluationWrapper + generators ────────────────
    def get_book(self):
        return self.book

    def get_df_qa(self):
        return self.df_qa

    def nb_tokens(self):
        return self._nb_tokens

    def nb_chapters(self):
        return self._nb_chapters

    def chunk_paragraphs(self, input_str, my_split='\n'):
        chunks = input_str.split(my_split)
        return [c.strip() for c in chunks if c.strip()]

    def chunk_book(self, split='chapter'):
        if split == 'chapter':
            return self.chunk_paragraphs(self.book, '\n\n\n')
        elif split == 'paragraph':
            xss = [[f"Chapter {k}, Paragraph {idx+1}\n\n{x}"
                    for idx, x in enumerate(self.chunk_paragraphs(v))]
                   for k, v in self.split_chapters.items()]
            return [x for xs in xss for x in xs]
