# Compresh arm — code

These files integrate a Compresh (TUL 2.0) answering arm into the upstream
EpBench harness. They are not standalone; they `import epbench.src.*` and are
meant to be dropped into a clone of
[ahstat/episodic-memory-benchmark](https://github.com/ahstat/episodic-memory-benchmark)
(see ../REPRODUCE.md).

| File | Role |
|---|---|
| `run_compresh.py` | Driver. Runs one arm (`--answering_kind compresh\|prompting\|rag`) on a chosen book, routes the **judge** through OpenRouter (default `gpt-4o`) instead of the benchmark builder, and sets the production retrieval policy (threshold 0.15 / rel_frac 0 / min_k 8). For `rag` it adds `--top_n` + the embedder. |
| `score_compresh_paper_method.py` | Scores the three arms with the benchmark's **own** ranking functions (Simple Recall = unweighted bin mean over recall questions; Chronological Awareness = the latest/exact-match/τ composite), so numbers match the upstream ranking. Reads cached answers — no API cost. |
| `generator_answers_4_compresh.py` | The Compresh arm itself: query-aware retrieval over chapters using the TUL 2.0 MiniLM engine, `cutoff = max(threshold, rel_frac·max_sim)` with a `min_k` floor, then the benchmark's standard answer/judge path. |
| `loaded_benchmark.py` | Loads the precomputed book + Q&A directly (drop-in for the generation wrapper) so scoring works under Python 3.14 / pandas 3.x without re-generating or re-asserting. |

The retrieval policy here mirrors what Compresh runs in production (TUL 2.0):
a stable, cacheable compressed **prefix** plus a small query-aware **suffix**
per turn.
