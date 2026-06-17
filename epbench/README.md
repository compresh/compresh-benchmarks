# EpBench — Compresh results

[EpBench](https://github.com/ahstat/episodic-memory-benchmark) (the Tulving
Episodic Memory Benchmark, ICLR 2025) measures how well a system can recall
*episodic* facts — who/what/where/when — from a long generated "book", and
whether it can place events in the right **chronological** order.

We add a **Compresh (TUL 2.0)** answering arm and compare it, on the same
model and the same independent judge, against two baselines the benchmark
already ships:

| Arm | What the model sees each question |
|---|---|
| **raw / full context** | the entire 196-chapter, ~103k-token book |
| **naive RAG (chapter, top-17)** | the 17 highest-similarity chapters |
| **Compresh · TUL 2.0** | a **query-aware slice** (our retrieval), never the whole book |

- **Book:** default universe, 200 events → 196 chapters / 102,870 tokens, seed 0.
- **Answerer:** `gpt-5-mini` for all three arms.
- **Judge:** an independent model (`gpt-4o-2024-08-06` via OpenRouter) — *not*
  the system under test. The benchmark's own judge gives the same model's raw
  Simple Recall as 0.830; ours gives 0.804 — within ~2pp, confirming the
  pipeline is consistent.
- **Scoring:** the benchmark's **own** functions
  (`epbench.src.evaluation.ranking`), so these numbers are directly comparable
  to the upstream ranking table.

## Simple Recall (paper method)

Unweighted mean over the five event-count bins (recall questions only):

| Arm (gpt-5-mini) | Simple Recall | Context read |
|---|---|---|
| **Compresh · TUL 2.0** | **0.828** | query-aware slice |
| raw / full context | 0.804 | 196 chapters |
| naive RAG · chapter (top-17) | 0.796 | 17 chapters |

Compresh has the highest Simple Recall **while never reading the whole book** —
and its margin grows on harder, multi-event questions:

| Matching events | count | raw | naive RAG | Compresh |
|---|---|---|---|---|
| 0 | 150 | 0.97 | 0.93 | 0.96 |
| 1 | 150 | 0.87 | 0.83 | 0.85 |
| 2 | 90 | 0.78 | 0.77 | **0.80** |
| 3–5 | 98 | 0.76 | 0.77 | **0.83** |
| 6+ | 60 | 0.64 | 0.68 | **0.70** |

## Chronological Awareness (paper method)

The benchmark's chronological score is a composite:
`mean(latest-state accuracy, %exact-set-match × Kendall-τ-on-exact-matches)`.

| Arm (gpt-5-mini) | Chronological Awareness | Latest | %exact-set | Kendall τ |
|---|---|---|---|---|
| **naive RAG · chapter (top-17)** | **0.651** | 86.0% | 51.3% | 0.86 |
| Compresh · TUL 2.0 | 0.442 | 60.2% | 28.2% | 1.00 |
| raw / full context | 0.428 | 62.9% | 23.1% | 0.98 |

**Honest read:** naive RAG leads on chronological ordering here. Reading 17
chapters gives it more raw material for "latest state" and for recovering the
*full* set of events to order — so it scores the composite higher. Compresh
orders **perfectly when it has the set** (τ = 1.00) but, by retrieving a tighter
query-aware slice, it captures the complete event set less often. Improving
recall of the full event set on ordering questions is on the TUL roadmap.

## Takeaways

- On an independent, published episodic benchmark, Compresh **edges out
  full-context recall** (0.828 vs 0.804) and beats naive top-k RAG (0.796) —
  without ever sending the whole book.
- The advantage concentrates where it matters: **multi-event recall**.
- On **chronological ordering**, naive RAG currently leads; we publish that
  rather than hide it.

## Reproduce

See [`REPRODUCE.md`](REPRODUCE.md). Data comes from the upstream EpBench repo
and its figshare release (not redistributed here). Attribution: [`NOTICE`](NOTICE).
