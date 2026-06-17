"""Compresh (TUL 2.0) on EpBench — answering + judge runner.

Mirrors quickstart.py but:
  (a) supports --answering_kind compresh  (our TUL 2.0 query-aware retrieval),
  (b) routes the JUDGE through OpenRouter (default gpt-4o-2024-08-06) instead of
      the Claude benchmark-builder — set via answering_parameters['judge_model'],
  (c) sets the production retrieval policy (threshold 0.15 / rel_frac 0 / min_k 8).

Data must already be present (figshare) — no generation/API cost there.
Answering (gpt-5-mini) → OpenAI key; judge (gpt-4o-or) → OpenRouter key. Both
read from --env_file.

Run from the EpBench repo root with the comp-proof venv active:
  # smoke (short book, ~cents):
  python epbench/experiments/run_compresh.py --book_nb_events 20 --answering_kind compresh
  # baseline raw (same judge) for comparison:
  python epbench/experiments/run_compresh.py --book_nb_events 20 --answering_kind prompting
  # full default-long:
  python epbench/experiments/run_compresh.py --book_nb_events 200 --answering_kind compresh
"""
from pathlib import Path
import argparse
import sys
# Make the repo root importable so `import epbench` works regardless of cwd.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

ap = argparse.ArgumentParser()
ap.add_argument('--data_folder', default='epbench/data')
ap.add_argument('--env_file', default=str(Path('~/projects/comp-work/.env').expanduser()))
ap.add_argument('--book_nb_events', type=int, default=20, help='20 = short (~10k), 200 = long (~100k)')
ap.add_argument('--answering_kind', default='compresh', help='compresh | prompting | rag')
ap.add_argument('--answering_model_name', default='gpt-5-mini')
ap.add_argument('--judge_model', default='openrouter/openai/gpt-4o-2024-08-06')
ap.add_argument('--retrieval_threshold', type=float, default=0.15)
ap.add_argument('--retrieval_min_k', type=int, default=8)
# RAG arm only (naive top-k baseline): chapters retrieved per question + embedder.
ap.add_argument('--top_n', type=int, default=17, help='RAG: chapters per question (matches published chapter-top17)')
ap.add_argument('--embedding_model', default='text-embedding-3-small')
ap.add_argument('--embedding_batch_size', type=int, default=2048)
args = ap.parse_args()

data_folder = Path(args.data_folder)
env_file = Path(args.env_file)

# Book identity — MUST match the precomputed figshare data (claude-3-5-sonnet,
# default universe/style, seed 0). This only selects which book to LOAD.
book_parameters = {'indexing': 'default', 'nb_summaries': 0}
prompt_parameters = {
    'nb_events': args.book_nb_events,
    'name_universe': 'default',
    'name_styles': 'default',
    'seed': 0,
    'distribution_events': {'name': 'geometric', 'param': 0.1},
}
model_parameters = {'model_name': 'claude-3-5-sonnet-20240620', 'max_new_tokens': 4096, 'itermax': 10}

from epbench.src.evaluation.loaded_benchmark import LoadedBenchmark
# Load precomputed book + Q&A directly — bypass the generation pipeline
# (BenchmarkGenerationWrapper.__end2end regenerates + asserts, which breaks
# under Python 3.14 / pandas 3.x). Data is final on figshare.
my_benchmark = LoadedBenchmark(prompt_parameters, model_parameters, book_parameters, data_folder)

answering_parameters = {
    'kind': args.answering_kind,
    'model_name': args.answering_model_name,   # answering model (gpt-5-mini → OpenAI)
    'max_new_tokens': 4096,
    'sleeping_time': 0,
    'policy': 'remove_duplicates',
    'judge_model': args.judge_model,           # judge (gpt-4o via OpenRouter)
    'embedding_chunk': 'chapter',
    'retrieval_threshold': args.retrieval_threshold,
    'retrieval_rel_frac': 0.0,
    'retrieval_min_k': args.retrieval_min_k,
    # RAG arm only (ignored by compresh/prompting):
    'top_n': args.top_n,
    'embedding_model': args.embedding_model,
    'embedding_batch_size': args.embedding_batch_size,
}

from epbench.src.evaluation.evaluation_wrapper import EvaluationWrapper
my_eval = EvaluationWrapper(my_benchmark, answering_parameters, data_folder, env_file)

print("\n=== Simple Recall (F1 lenient, by #matching events) ===")
print(my_eval.get_pretty_summary_relative_to(
    my_column='bins_items_correct_answer', metric='f1_score_lenient'))
print("\n=== Chronological awareness ===")
print(my_eval.kendall_summaries_for_this_experiment)
print(f"\nkind={args.answering_kind} answer={args.answering_model_name} judge={args.judge_model} "
      f"book_events={args.book_nb_events}")
print('Ended successfully')
