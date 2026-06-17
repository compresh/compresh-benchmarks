"""Score our three gpt-5-mini arms (raw / RAG / Compresh) with EpBench's OWN
paper-method scoring — so the numbers are directly comparable to the published
ranking (e.g. gpt-5-mini raw = 0.830 Simple Recall / 0.442 Chronological).

Why this exists: hand-rolled means do NOT reproduce the paper's headline scores.
  - Simple Recall  = UNWEIGHTED mean over the 5 event-count bins, get=='all' only.
  - Chronological  = mean(Latest-state acc, %exact-match × Kendall-τ-on-exact).
Both are produced by epbench.src.evaluation.ranking. We just feed it our arms.

No API cost: answers/evaluations/chronological are read from cache on disk.
Uses LoadedBenchmark to skip the fragile generation pipeline (Py3.14/pandas3).

Run from the EpBench repo root, comp-proof venv active:
  python epbench/experiments/score_compresh_paper_method.py --env_file .env
"""
from pathlib import Path
import argparse
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument('--data_folder', default='epbench/data')
ap.add_argument('--env_file', default='.env')
ap.add_argument('--book_nb_events', type=int, default=200)
ap.add_argument('--judge_model', default='openrouter/openai/gpt-4o-2024-08-06')
ap.add_argument('--rag_top_n', type=int, default=17)  # must match the cached RAG run
args = ap.parse_args()

data_folder = Path(args.data_folder)
env_file = Path(args.env_file)

from epbench.src.evaluation.loaded_benchmark import LoadedBenchmark
from epbench.src.evaluation.evaluation_wrapper import EvaluationWrapper
from epbench.src.evaluation.ranking import (
    get_simple_results, simple_recall_score,
    get_kendall_tau_results, chronological_awareness,
)

book_parameters = {'indexing': 'default', 'nb_summaries': 0}
prompt_parameters = {'nb_events': args.book_nb_events, 'name_universe': 'default',
                     'name_styles': 'default', 'seed': 0,
                     'distribution_events': {'name': 'geometric', 'param': 0.1}}
model_parameters = {'model_name': 'claude-3-5-sonnet-20240620', 'max_new_tokens': 4096, 'itermax': 10}

bench = LoadedBenchmark(prompt_parameters, model_parameters, book_parameters, data_folder)

# Our three arms, all gpt-5-mini, same OpenRouter judge — read from cache.
arms = [
    {'kind': 'prompting', 'chunk': 'n/a'},
    {'kind': 'rag',       'chunk': 'chapter'},
    {'kind': 'compresh',  'chunk': 'chapter'},
]

rows = []
for a in arms:
    ap_ = {'kind': a['kind'], 'model_name': 'gpt-5-mini', 'max_new_tokens': 4096,
           'sleeping_time': 0, 'policy': 'remove_duplicates', 'judge_model': args.judge_model}
    if a['kind'] == 'rag':
        ap_.update({'embedding_chunk': 'chapter', 'embedding_model': 'text-embedding-3-small',
                    'embedding_batch_size': 2048, 'top_n': args.rag_top_n})
    if a['kind'] == 'compresh':
        ap_.update({'embedding_chunk': 'chapter', 'retrieval_threshold': 0.15,
                    'retrieval_rel_frac': 0.0, 'retrieval_min_k': 8})
    ev = EvaluationWrapper(bench, ap_, data_folder, env_file)
    rows.append({'book_nb_events': args.book_nb_events,
                 'answering_kind': a['kind'],
                 'answering_model_name': 'gpt-5-mini',
                 'answering_embedding_chunk': a['chunk'],
                 'book_model_name': 'claude-3-5-sonnet-20240620',
                 'evaluation_object': ev})

df = pd.DataFrame(rows)

ne = args.book_nb_events
print("\n================ Simple Recall — per bin (get=='all') ================")
print(get_simple_results(df, ne).to_string())
print("\n================ Simple Recall SCORE (paper method) ==================")
print(simple_recall_score(get_simple_results(df, ne)).to_string())

print("\n================ Chronological — Latest / All / Kendall τ ============")
ktr = get_kendall_tau_results(df, ne)
print(ktr.to_string())
print("\n================ Chronological Awareness SCORE (paper method) ========")
print(chronological_awareness(ktr).to_string())

print("\nReference (paper README, gpt-5-mini raw): Simple Recall 0.830 · Chronological 0.442")
print("Ended successfully")
