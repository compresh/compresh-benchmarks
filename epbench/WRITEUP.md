# Fewer tokens, same recall: reconstruct context, don't resend it

Most LLM apps do the same thing every turn: they resend the entire conversation. The transcript grows linearly,
each turn costs more than the last, and — past a point — the model gets *worse*, not better, because long
context degrades (the "lost in the middle" effect, and what people now call context rot: quality drops well
before the nominal window is full).

Compresh takes a different path. Instead of resending the whole history, it **reconstructs** a query-aware slice
of it each turn — the part of the past this turn actually needs. The obvious question is whether recall survives
when you stop sending the whole thing. So we measured it on an independent benchmark, and we publish where it
wins **and** where it loses.

## The axis: savings × quality, not recall alone

Most agent-memory work optimizes one number: recall or accuracy. We care about a different one — **how few
tokens you can send while holding quality.** Two measurements:

- **Compression.** On 360 real StackExchange Q&A items, replayed as one long, growing session, our open-source
  core ([tulbase](https://github.com/compresh/compresh)) sent **66% fewer input tokens** (40.9M → 13.9M) with
  no measurable quality loss (answer equivalence 87.5% vs 90.0% raw; cosine 0.667 vs 0.670).
- **Reconstruction (the paid memory layer, TUL 2.0).** On a strong model, a single turn goes from **31,947 →
  275 input tokens (−99.1%)** — it sends a query-aware slice, not the conversation. (The system prompt is left
  untouched.)

Fewer tokens is easy if you don't care about answers. The point is holding quality — so here's the benchmark.

## The benchmark

We used **EpBench** — an independent, published episodic-memory benchmark (ICLR 2025; built on Tulving's model
of recall): cued questions over a long, generated book. Same answerer (gpt-5-mini) and the same judge across
every arm, scored with **the benchmark's own method** — no home-field scoring.

| Method                 | Simple recall | Context read |
| ---------------------- | ------------- | ------------ |
| raw / full context     | 0.804         | 196 chapters |
| naive RAG · chapter    | 0.796         | 17 chapters  |
| **Compresh · TUL 2.0** | **0.828**     | query-aware  |

The point is the juxtaposition — recall is essentially at parity while tokens are not:

```
EpBench · Simple Recall (paper method) · gpt-5-mini
──────────────────────────────────────────────────────
  Compresh · TUL 2.0  0.828 [█████████████████░░░]  query-aware slice
  raw / full context  0.804 [████████████████░░░░]  196 chapters
  naive RAG · top-17   0.796 [████████████████░░░░]  17 chapters

Input tokens / turn (strong model, long chat)
──────────────────────────────────────────────────────
  raw                31,947 [████████████████████]
  Compresh              275 [▏░░░░░░░░░░░░░░░░░░░]  −99.1%
```

Compresh has the highest simple recall **while reading a query-aware slice, not the whole ~103k-token book** —
and pulls further ahead on multi-event questions (full per-bin breakdown in
[`results/`](results/epbench_gpt5mini_simple_recall_by_bin.csv)). Judge caveat, stated up front: our judge was
OpenRouter gpt-4o; the paper's own judge puts raw at 0.830 — within ~2 points. Same judge for all arms.

You can reproduce the headline in ~10 seconds, no API keys: [`verify.py`](../verify.py) recomputes Simple Recall
(the paper method — an unweighted mean over the matching-event bins) from the published per-bin recalls and
checks it against the scoreboard.

## Where it loses — and why that's the honest part

On **chronological ordering**, naive RAG beats us: **0.65 vs 0.44.** Retrieving a query-relevant slice breaks
temporal contiguity, so "put these events in order" gets harder. We publish that number next to the wins.

This isn't a confession of inferiority — it's the nature of the field. **Every approach here trades something.**
Long context keeps everything and loses the middle. RAG retrieves by similarity and loses coherence and order.
Summarization keeps a gist and loses detail. Reconstruction keeps what the turn needs and (today) loses some
chronology. Loss is already everywhere in context and memory systems; the only real choice is whether you
measure it and say so. We did, and we published both sides.

## "But what about prefix caching?"

A fair objection: you don't have to recompute a stable prefix — providers cache it. True, and prefix caching is
a real, powerful serving optimization. But it's worth being precise about what it does and doesn't do:

- It makes **resending a lot** cheaper to serve. It does **not** make the history smaller — you still ship the
  whole transcript every turn, just at a discount on the cached part.
- It does **nothing** for the quality problem. Lost-in-the-middle degradation is orthogonal to caching: a
  perfectly cached 100k-token context still loses the middle. So the recall result above stands regardless.

In other words, prefix caching optimizes the **symptom** (recompute cost), not the **cause** (you're sending too
much). And cached tokens still aren't free — roughly 10–50% of base input price depending on provider.

So the honest cost comparison isn't "Compresh vs raw-without-caching." It's **Compresh vs raw + prefix caching**.
Modeling that — generously to the cached baseline (assuming a full cache hit every turn, ignoring cache-write
premiums and TTL misses) — the crossover is around **~10k tokens of history.** Below that, raw+cache can be
cheaper: Compresh has a small fixed per-turn overhead. Above it, Compresh wins, and the gap **widens as the
conversation grows**, because raw scales with length while the reconstructed slice stays roughly flat. (This is
a modeled result; a live-capture confirmation is in progress.) The takeaway is honest and narrow: **this is a
long-conversation argument**, not a "cheaper for everything" one.

## How it works (briefly)

Each turn, Compresh takes the full history, builds a query-aware reconstruction of the older part
(`compresh_md`), and keeps a protected tail of recent raw turns (`raw_tail`). The model receives
`compresh_md + raw_tail` instead of the full transcript. It differs from RAG — we reconstruct the
*conversation* per turn, not retrieve documents — and from prompt compression like LLMLingua — we don't drop
tokens by perplexity; we rebuild the query-relevant history. The system prompt is never compressed.

## Reproduce it / try it

- **Verify the headline (~10s, no keys):** [`../verify.py`](../verify.py)
- **Full re-run (calls the models + an independent judge, needs keys):** [`REPRODUCE.md`](REPRODUCE.md)
- **Try Compresh:** one line — change your `base_url`, keep everything else. Free to start, no card, pay only on
  the tokens it removes: [compre.sh](https://compre.sh)

We'd genuinely value pushback on the method and the cost model.
