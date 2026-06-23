#!/usr/bin/env python3
"""
verify.py — reproduce the EpBench headline numbers in ~10 seconds, no API keys.

It recomputes Simple Recall (paper method = unweighted mean over the matching-event
bins) from the published per-bin recalls, and checks it against the published
headline scoreboard. Pure standard library — no install, no keys, no network.

    python3 verify.py

For the FULL re-run (calls models + judge, needs keys) see epbench/REPRODUCE.md.
"""
import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BY_BIN = os.path.join(HERE, "epbench", "results", "epbench_gpt5mini_simple_recall_by_bin.csv")
PAPER = os.path.join(HERE, "epbench", "results", "epbench_gpt5mini_paper_method.csv")

# map the by-bin column name -> the method label used in the paper_method.csv
COLS = {
    "raw": "raw / full context",
    "naive_rag_chapter_top17": "naive RAG (chapter top17)",
    "compresh_tul2": "Compresh TUL 2.0 (query-aware)",
}


def load_bins(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def load_paper(path):
    out = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            out[r["method"]] = r
    return out


def main():
    bins = load_bins(BY_BIN)
    paper = load_paper(PAPER)

    print("EpBench — recomputing the headline from published per-bin recalls (no API keys)\n")
    print("Per-bin Simple Recall (paper method = UNWEIGHTED mean over these bins):")
    disp = {"raw": "raw", "naive_rag_chapter_top17": "naiveRAG", "compresh_tul2": "compresh"}
    hdr = f"  {'bin':<5}{'n':>5}   " + "".join(f"{disp[c]:>16}" for c in COLS)
    print(hdr)
    for row in bins:
        line = f"  {row['bin_matching_events']:<5}{row['count']:>5}   "
        line += "".join(f"{float(row[c]):>16.3f}" for c in COLS)
        print(line)

    print("\nHeadline check — unweighted bin-mean vs published simple_recall_paper_method:")
    print(f"  {'method':<34}{'computed':>10}{'published':>11}{'match':>8}")
    ok = 0
    for col, label in COLS.items():
        computed = sum(float(r[col]) for r in bins) / len(bins)
        published = float(paper[label]["simple_recall_paper_method"])
        match = abs(computed - published) < 1e-3
        ok += match
        print(f"  {label:<34}{computed:>10.3f}{published:>11.3f}{('OK' if match else 'FAIL'):>8}")

    print("\nFull paper-method scoreboard (results/epbench_gpt5mini_paper_method.csv):")
    print(f"  {'method':<46}{'simple_recall':>14}{'chronological':>15}")
    for label, r in paper.items():
        print(f"  {label:<46}{r['simple_recall_paper_method']:>14}{r['chronological_awareness_paper_method']:>15}")

    print("\nNotes:")
    print("  - All arms: same answerer (gpt-5-mini) + same judge (OpenRouter gpt-4o);")
    print("    the paper's own judge puts raw at 0.830 (within ~2pp).")
    print("  - chronological_awareness is a composite (paper method) computed by")
    print("    epbench/code/score_compresh_paper_method.py; full re-run in epbench/REPRODUCE.md.")
    print(f"\nRESULT: {ok}/{len(COLS)} headline numbers reproduce from the published per-bin data.")
    return 0 if ok == len(COLS) else 1


if __name__ == "__main__":
    sys.exit(main())
