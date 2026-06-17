"""Compresh (TUL 2.0) answering arm for EpBench.

Query-aware retrieval over the book chapters, identical to the production
proxy (proxy/optimizer.retrieve_history) and to the LongMemEval runner:
MiniLM cosine + relative-threshold + min-K floor (production policy:
threshold=0.15, rel_frac=0.0, min_k=8). Unlike the `rag` arm it does NOT use
OpenAI embeddings — it reuses the SAME local MiniLM embedder as our other
benchmarks (LongMemEval/engine/tulngin) so retrieval is byte-identical across
benches. Falls back to sentence-transformers all-MiniLM-L6-v2 if the engine
import fails.

Selected chapters are rendered into the same prompt as the `rag` arm
(generate_episodic_memory_rag_prompt), so answering + judging are unchanged.
"""
from pathlib import Path
import sys
import numpy as np

from epbench.src.evaluation.generator_answers_2_rag import generate_episodic_memory_rag_prompt

# ── Embedder: reuse the production/LongMemEval tulngin MiniLM ────────────────
_ENGINE_DIR = Path("~/projects/comp-work/comp-research/Benchmarks/LongMemEval/engine").expanduser()
_embed_pool = None   # batch encode (chunks) -> [N, d] normalized
_embed_np = None     # single encode (query) -> [d] normalized
_ST_MODEL = None     # sentence-transformers fallback


def _ensure_embedder():
    global _embed_pool, _embed_np, _ST_MODEL
    if _embed_pool is not None or _ST_MODEL is not None:
        return
    if str(_ENGINE_DIR) not in sys.path:
        sys.path.insert(0, str(_ENGINE_DIR))
    try:
        from tulngin.mmr_compose import _embed_pool as ep   # type: ignore
        from tulngin.semantic_store import _embed_np as en   # type: ignore
        _embed_pool, _embed_np = ep, en
    except Exception as e:
        print(f"[compresh] tulngin engine import failed ({e!r}); "
              f"falling back to sentence-transformers all-MiniLM-L6-v2")
        from sentence_transformers import SentenceTransformer
        _ST_MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def _embed_chunks(texts):
    _ensure_embedder()
    if _embed_pool is not None:
        return np.asarray(_embed_pool(list(texts)), dtype=np.float32)
    v = _ST_MODEL.encode(list(texts), normalize_embeddings=True,
                         convert_to_numpy=True, show_progress_bar=False)
    return np.asarray(v, dtype=np.float32)


def _embed_query(text):
    _ensure_embedder()
    if _embed_np is not None:
        return np.asarray(_embed_np(text), dtype=np.float32)
    v = _ST_MODEL.encode([text], normalize_embeddings=True,
                         convert_to_numpy=True, show_progress_bar=False)
    return np.asarray(v[0], dtype=np.float32)


# Per-process cache: chunks identity -> embedding matrix (embed once per book).
_CACHE = {}


def query_message_compresh(question, chunks, answering_parameters, env_file=None):
    """Build the answering prompt from Compresh-retrieved chapters.

    Same selection as proxy/optimizer.retrieve_history:
      cutoff = max(threshold, rel_frac * max_sim); always keep top min_k.
    """
    threshold = float(answering_parameters.get('retrieval_threshold', 0.15))
    rel_frac = float(answering_parameters.get('retrieval_rel_frac', 0.0))
    min_k = int(answering_parameters.get('retrieval_min_k', 8))

    chunks = list(chunks)
    n = len(chunks)
    if n == 0:
        return generate_episodic_memory_rag_prompt("", question)

    # Embed each book once (chapters are reused across all its questions).
    cache_key = (n, hash(chunks[0]), hash(chunks[-1]))
    E = _CACHE.get(cache_key)
    if E is None:
        E = _embed_chunks(chunks)
        _CACHE[cache_key] = E

    qv = _embed_query(question)
    sims = E @ qv  # cosine (both unit-norm)

    order = sorted(range(n), key=lambda j: float(sims[j]), reverse=True)
    max_sim = float(sims[order[0]]) if order else -1.0
    cutoff = max(threshold, rel_frac * max_sim) if max_sim > 0 else -1.0

    keep = set(order[:max(0, min(min_k, n))])          # min-K floor
    for j in order:                                     # + (relative/absolute) cutoff
        if float(sims[j]) >= cutoff and float(sims[j]) > 0:
            keep.add(j)
    keep = sorted(keep)                                 # chronological render

    book_content = ""
    for j in keep:
        book_content += f'\n\n"""\n{chunks[j]}\n"""'
    return generate_episodic_memory_rag_prompt(book_content, question)
