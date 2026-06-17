# compresh-benchmarks

Public, reproducible benchmark results for [Compresh](https://compre.sh) — the
context-compression and episodic-memory layer for LLM APIs.

We publish what we measure, including where our own systems fall short. Every
number here is produced with each benchmark's **own** scoring and graded by an
**independent judge** (not the system under test), so results are comparable to
the original benchmarks and reproducible from the published harnesses.

## Benchmarks

| Folder | Benchmark | Status |
|---|---|---|
| [`epbench/`](epbench/) | EpBench — the Tulving Episodic Memory Benchmark (ICLR 2025) | published |
| `comp-proof/` | Compression fidelity on long multi-turn sessions (StackExchange replay) | coming |
| `tbench/` | T-bench — episodic memory across answerer models | coming |

## Headline — EpBench (gpt-5-mini, 200-event book)

Same answerer and judge across arms; the benchmark's own scoring:

| Arm | Simple Recall | Context read |
|---|---|---|
| **Compresh · TUL 2.0** | **0.828** | query-aware slice |
| raw / full context | 0.804 | 196 chapters |
| naive RAG · chapter (top-17) | 0.796 | 17 chapters |

Compresh has the highest Simple Recall **without reading the whole book**, and
pulls further ahead on multi-event questions. On chronological *ordering*, naive
RAG currently leads (0.65 vs 0.44) — full breakdown and honest discussion in
[`epbench/`](epbench/).

## Method, in one line

Independent published benchmarks · independent judge · each benchmark's own
scoring · open harness + reproduction steps. See each folder's `REPRODUCE.md`.

## License & attribution

Our code and analysis: MIT, © 2026 Compresh Ltd ([`LICENSE`](LICENSE)).
Third-party benchmarks keep their own licenses and authorship — see each
folder's `NOTICE` (e.g. [`epbench/NOTICE`](epbench/NOTICE)).
