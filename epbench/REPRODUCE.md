# Reproducing the EpBench results

These results run on top of the upstream EpBench harness. We do not redistribute
the dataset; you bring it from upstream.

## 1. Get the harness + data

```bash
git clone https://github.com/ahstat/episodic-memory-benchmark
cd episodic-memory-benchmark
# download the figshare data release linked in the upstream README into epbench/data/
```

## 2. Add the Compresh arm

Copy the files from `code/` into the clone:

| File | Destination in the clone |
|---|---|
| `run_compresh.py` | `epbench/experiments/` |
| `score_compresh_paper_method.py` | `epbench/experiments/` |
| `generator_answers_4_compresh.py` | `epbench/src/evaluation/` |
| `loaded_benchmark.py` | `epbench/src/evaluation/` |

`loaded_benchmark.py` loads the precomputed book + Q&A directly and bypasses the
generation pipeline (which is fragile under Python 3.14 / pandas 3.x).

Small edits are also needed in upstream files (a `compresh` branch and an
OpenRouter judge route). The touched files are:
`evaluation_wrapper.py`, `generator_answers_1_prompting.py`, `models_wrapper.py`,
`io.py`, `settings_wrapper.py`, `scoring_answers.py`, `ranking.py`,
`results/average_groups.py`. Each change is additive (a new branch / an optional
import); see the inline comments in `code/` and CODE.md.

## 3. Environment

A minimal `.env` with only the keys the run needs:

```
OPENAI_API_KEY=...        # answering (gpt-5-mini)
OPENROUTER_API_KEY=...    # judge (gpt-4o), and gpt-5-mini if routed via OpenRouter
```

## 4. Run the three arms (200-event book)

```bash
# Compresh (TUL 2.0, query-aware retrieval)
python epbench/experiments/run_compresh.py --book_nb_events 200 --answering_kind compresh --env_file .env
# raw / full context
python epbench/experiments/run_compresh.py --book_nb_events 200 --answering_kind prompting --env_file .env
# naive RAG (chapter, top-17)
python epbench/experiments/run_compresh.py --book_nb_events 200 --answering_kind rag --top_n 17 --env_file .env
```

Answers, evaluations, and chronological judgments are cached on disk, so re-runs
cost nothing.

## 5. Score with the benchmark's own method

```bash
python epbench/experiments/score_compresh_paper_method.py --env_file .env
```

This prints Simple Recall and Chronological Awareness using the upstream
`epbench.src.evaluation.ranking` functions — the same scoring behind the
published ranking table — for all three arms.
