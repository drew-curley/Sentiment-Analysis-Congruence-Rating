#!/usr/bin/env python3
"""amazon.py

Two functions in one script:

1) Download Amazon Reviews 2023 (Books) via Hugging Face Datasets and write a CSV
2) Analyze the CSV to flag sarcasm candidates: positive-sounding text + low rating

Examples:
  # Download
  python amazon.py download --category raw_review_Books --out amazon_reviews_2023_books.csv

  # Analyze (fast VADER)
  python amazon.py analyze --input amazon_reviews_2023_books.csv --output sarcasm_candidates.csv --require-contrast
"""

import argparse
import re
import sys
from pathlib import Path

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


CONTRAST_RE = re.compile(r"\b(but|however|although|though|yet|except|still|nevertheless|nonetheless)\b", re.I)
SARCASM_MARKERS_RE = re.compile(
    r"\b(yeah\s+right|as\s+if|sure\s+\w+|what\s+a\s+(great|wonderful|amazing))\b",
    re.I,
)


def safe_str(x) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and pd.isna(x):
        return ""
    return str(x)


def compute_vader_compound(analyzer: SentimentIntensityAnalyzer, title: str, text: str) -> float:
    # Title often contains the punchline; give it a bit more weight by placing first.
    combined = (title.strip() + "\n\n" + text.strip()).strip()
    if not combined:
        return 0.0
    return analyzer.polarity_scores(combined)["compound"]


def chunk_reader(path: Path, chunksize: int):
    # pyarrow can be faster/more memory-friendly when installed; fall back gracefully.
    try:
        return pd.read_csv(path, chunksize=chunksize, engine="pyarrow")
    except Exception:
        return pd.read_csv(path, chunksize=chunksize)


def cmd_download(args: argparse.Namespace) -> None:
    """Download a HF dataset split and write to CSV."""
    try:
        from datasets import load_dataset  # lazy import so analyze-only doesn't need datasets
    except Exception as e:
        print(
            "Missing dependency 'datasets'. Install it with: pip install datasets\n"
            f"Original error: {e}",
            file=sys.stderr,
        )
        raise SystemExit(1)

    out_path = Path(args.out)
    if out_path.exists() and not args.force:
        print(f"Refusing to overwrite existing file: {out_path} (use --force)", file=sys.stderr)
        raise SystemExit(1)

    print(f"Loading dataset: McAuley-Lab/Amazon-Reviews-2023 | {args.category} | split={args.split}")
    ds = load_dataset("McAuley-Lab/Amazon-Reviews-2023", args.category, split=args.split)

    df = pd.DataFrame(ds)
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df):,} rows → {out_path.resolve()}")


def cmd_analyze(args: argparse.Namespace) -> None:
    in_path = Path(args.input)
    if not in_path.exists():
        print(f"Input not found: {in_path}", file=sys.stderr)
        raise SystemExit(1)

    analyzer = SentimentIntensityAnalyzer()

    out_path = Path(args.output)
    if out_path.exists():
        out_path.unlink()

    total = 0
    kept = 0

    required_cols = [
        "rating",
        "title",
        "text",
        "review_id",
        "asin",
        "user_id",
        "verified_purchase",
        "helpful_vote",
        "timestamp",
    ]

    for chunk in chunk_reader(in_path, args.chunksize):
        total += len(chunk)

        # Ensure required columns exist
        for col in required_cols:
            if col not in chunk.columns:
                chunk[col] = None

        # Low-rating filter first
        ratings = pd.to_numeric(chunk["rating"], errors="coerce").fillna(0).astype(int)
        chunk = chunk[ratings <= args.low_rating_max]
        if chunk.empty:
            continue

        # Build combined text (for regex checks + length filter)
        title_s = chunk["title"].map(safe_str)
        text_s = chunk["text"].map(safe_str)
        combined = (title_s.str.strip() + " " + text_s.str.strip()).str.strip()

        # Length filter
        chunk = chunk[combined.str.len() >= args.min_text_len]
        if chunk.empty:
            continue

        # Sentiment score (VADER)
        chunk["sent_compound"] = [
            compute_vader_compound(analyzer, t, x) for t, x in zip(title_s.loc[chunk.index], text_s.loc[chunk.index])
        ]

        # Contrast / markers
        comb_i = combined.loc[chunk.index]
        chunk["has_contrast"] = comb_i.str.contains(CONTRAST_RE, na=False)
        chunk["has_sarcasm_markers"] = comb_i.str.contains(SARCASM_MARKERS_RE, na=False)

        # Candidate rule
        is_positive = chunk["sent_compound"] >= args.pos_sentiment_min
        is_very_positive = chunk["sent_compound"] >= args.very_pos_sentiment_min

        if args.require_contrast:
            passes_structure = chunk["has_contrast"] | is_very_positive
        else:
            passes_structure = pd.Series(True, index=chunk.index)

        if args.use_sarcasm_markers:
            passes_structure = passes_structure | chunk["has_sarcasm_markers"]

        candidates = chunk[is_positive & passes_structure].copy()
        if candidates.empty:
            continue

        # Sarcasm mismatch score: sentiment minus normalized rating
        rating_norm = pd.to_numeric(candidates["rating"], errors="coerce").fillna(0) / 5.0
        candidates["sarcasm_score"] = candidates["sent_compound"] - rating_norm

        keep_cols = [
            "review_id",
            "asin",
            "user_id",
            "rating",
            "verified_purchase",
            "helpful_vote",
            "timestamp",
            "sent_compound",
            "sarcasm_score",
            "has_contrast",
            "has_sarcasm_markers",
            "title",
            "text",
        ]
        keep_cols = [c for c in keep_cols if c in candidates.columns]
        candidates = candidates[keep_cols]

        # Append
        candidates.to_csv(out_path, mode="a", index=False, header=not out_path.exists())
        kept += len(candidates)

        if args.progress_every > 0 and total % args.progress_every == 0:
            print(f"Processed {total:,} rows | candidates {kept:,}")

    print("\nDone.")
    print(f"Processed rows: {total:,}")
    print(f"Sarcasm candidates: {kept:,}")
    print(f"Output: {out_path.resolve()}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Download and analyze Amazon Reviews 2023 (Books) for sarcasm candidates.")
    sub = p.add_subparsers(dest="cmd", required=True)

    # download
    pdn = sub.add_parser("download", help="Download HF dataset to CSV")
    pdn.add_argument("--category", default="raw_review_Books", help="HF config/category, e.g. raw_review_Books")
    pdn.add_argument("--split", default="full", help="Dataset split (default: full)")
    pdn.add_argument("--out", default="amazon_reviews_2023_books.csv", help="Output CSV path")
    pdn.add_argument("--force", action="store_true", help="Overwrite output if it exists")
    pdn.set_defaults(func=cmd_download)

    # analyze
    pan = sub.add_parser("analyze", help="Analyze CSV for sarcasm candidates")
    pan.add_argument("--input", required=True, help="Input CSV file")
    pan.add_argument("--output", default="sarcasm_candidates.csv", help="Output CSV for candidates")
    pan.add_argument("--chunksize", type=int, default=100_000, help="Rows per chunk")
    pan.add_argument("--progress-every", type=int, default=500_000, help="Print progress every N rows (0 disables)")

    pan.add_argument("--low-rating-max", type=int, default=2, help="Low rating threshold (<= this)")
    pan.add_argument("--pos-sentiment-min", type=float, default=0.35, help="Min VADER compound to count as positive")
    pan.add_argument(
        "--very-pos-sentiment-min",
        type=float,
        default=0.55,
        help="If sentiment is this high, pass even without contrast terms",
    )

    pan.add_argument("--require-contrast", action="store_true", help="Require contrast markers unless very positive")
    pan.add_argument("--use-sarcasm-markers", action="store_true", help="Allow explicit sarcasm marker regex")
    pan.add_argument("--min-text-len", type=int, default=40, help="Min chars of (title+text)")
    pan.set_defaults(func=cmd_analyze)

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()