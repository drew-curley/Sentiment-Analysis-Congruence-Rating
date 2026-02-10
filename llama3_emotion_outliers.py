#!/usr/bin/env python3
"""
Emotionally charged sentence outliers using Llama 3 via Ollama (Option B).

Supports:
- .txt input (entire file)
- .csv input (specified text column)

Sentence splitting:
- Default: regex-based splitter (no heavy deps)
- Optional: spaCy splitter if installed and selected

Outputs:
- flagged_sentences.csv with robust MAD z-scores
"""

import argparse
import json
import re
import time
from typing import Any, Dict, List, Optional, Tuple

# ----------------------------
# Gutenberg / mojibake cleanup
# ----------------------------
# Common mojibake sequences found in Gutenberg-style texts.
# We prefer to normalize common punctuation to ASCII/Unicode equivalents,
# then remove any remaining known artifact tokens.
_GUTENBERG_REPLACEMENTS = {
    # Windows-1252/UTF-8 mojibake for quotes/dashes/ellipsis
    "‚Äô": "'",   # right single quote
    "‚Äò": "'",   # left single quote
    "‚Äù": '"',  # right double quote
    "‚Äú": '"',  # left double quote
    "‚Äî": "—",  # em dash
    "‚Äì": "–",  # en dash
    "‚Ä¶": "...", # ellipsis
    "‚Ä¢": "•",  # bullet

    # Another common mojibake family
    "â€™": "'",
    "â€˜": "'",
    "â€œ": '"',
    "â€�": '"',
    "â€”": "—",
    "â€“": "–",
    "â€¦": "...",

    # Non-breaking space artifacts
    "\u00a0": " ",
}

# Tokens we simply strip (leave empty) because they are rarely meaningful
# in English corpora and often indicate encoding corruption.
_GUTENBERG_STRIP_TOKENS = [
    "Äù",  # seen in some Gutenberg conversions
    "Â",   # stray NBSP marker
    "�",   # Unicode replacement char
    # Occasionally the mojibake introducers appear alone
    "‚", "Ä",
    # Common double-encoded Latin-1 fragments (safe to strip in English data)
    "Ã©", "Ã¨", "Ã", "Ã¶", "Ã¼", "Ã±",
]


def strip_gutenberg_artifacts(text: str) -> str:
    # First normalize known multi-character mojibake sequences
    for bad, good in _GUTENBERG_REPLACEMENTS.items():
        text = text.replace(bad, good)

    # Then strip remaining known tokens
    for tok in _GUTENBERG_STRIP_TOKENS:
        text = text.replace(tok, "")

    # Final light cleanup: collapse repeated spaces introduced by removals
    text = re.sub(r"\s+", " ", text).strip()

    # Normalize common “smart” punctuation to ASCII equivalents.
    # This removes curly quotes/dashes that can still appear even after mojibake fixes.
    text = (text
            .replace("’", "'")
            .replace("‘", "'")
            .replace("“", '"')
            .replace("”", '"')
            .replace("—", "-")
            .replace("–", "-")
            .replace("…", "...")
    )

    # Preserve legitimate non-ASCII letters/diacritics (e.g., in names), but remove stray control characters.
    # The mojibake-specific replacements/strips above handle the problematic sequences.
    text = "".join(ch for ch in text if ch.isprintable())

    # Re-collapse whitespace in case normalization introduced spacing changes
    text = re.sub(r"\s+", " ", text).strip()
    return text

import numpy as np
import pandas as pd
import requests
# ----------------------------
# Sentence splitting (regex fallback)
# ----------------------------
_ABBREV = {
    "mr.", "mrs.", "ms.", "dr.", "prof.", "sr.", "jr.",
    "st.", "vs.", "etc.", "e.g.", "i.e.", "fig.", "no.",
    "u.s.", "u.k.", "inc.", "ltd.",
}


def split_sentences_regex(text: str) -> List[str]:
    """Naive but decent sentence splitter that avoids external libs.

    Splits on ., !, ? followed by whitespace/newline, while trying to avoid common abbreviations.
    """
    text = re.sub(r"\s+", " ", text.strip())
    if not text:
        return []

    # Candidate split positions
    parts: List[str] = []
    start = 0
    for m in re.finditer(r"[.!?]", text):
        end = m.end()
        # Look ahead: must be end-of-string or whitespace
        if end < len(text) and text[end] not in " \t\n\r":
            continue

        chunk = text[start:end].strip()
        if chunk:
            last_word = chunk.split()[-1].lower()
            # Avoid splitting after abbreviations like "Dr." or "U.S."
            if last_word in _ABBREV:
                continue
            parts.append(chunk)
            start = end

    tail = text[start:].strip()
    if tail:
        parts.append(tail)

    return [p.strip() for p in parts if p.strip()]


# ----------------------------
# Robust z-score (MAD)
# ----------------------------
def robust_z_mad(x: np.ndarray) -> np.ndarray:
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    denom = 1.4826 * mad if mad > 0 else 1e-9
    return (x - med) / denom


# ----------------------------
# Ollama client
# ----------------------------
def ollama_generate(
    prompt: str,
    *,
    model: str = "llama3",
    host: str = "http://localhost:11434",
    temperature: float = 0.0,
    top_p: float = 1.0,
    num_predict: int = 120,
    timeout: int = 120,
) -> str:
    url = f"{host}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "top_p": top_p,
            "num_predict": num_predict,
        },
    }
    r = requests.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json().get("response", "")


# ----------------------------
# Prompt
# ----------------------------
def build_prompt(sentence: str) -> str:
    return f"""
You are a scoring function. Output ONLY valid JSON.

Task: Given ONE sentence, rate EMOTIONAL INTENSITY of the language itself on a 0–10 scale.

Definition:
- Emotional intensity = strong feeling, urgency, distress, outrage, fear, desperation, or emphatic judgment in wording.
- Focus on emotionally loaded adjectives/adverbs, intensifiers, threats, absolute terms.
- Do NOT infer emotion from topic alone.
- Ignore sarcasm unless clearly intense.

Scale anchors:
0 = neutral, factual
3 = mild evaluative language
6 = clearly emotional
9 = extreme emotion

Return JSON:
{{
  "intensity": number,
  "valence": number,
  "spans": [
    {{"text":"...", "start": int, "end": int, "type":"ADJ|ADV|VERB|PHRASE|INTENSIFIER|THREAT", "weight": number}}
  ],
  "notes": "optional"
}}

Valence definition:
- valence is the emotional direction of the language itself on a -1 to +1 scale.
  -1 = strongly negative (fear, disgust, anger, despair)
   0 = neutral / no clear direction
  +1 = strongly positive (joy, relief, admiration, hope)
- If language is emotionally intense but direction is mixed, choose the dominant direction; if truly mixed/unclear, use 0.
"""

def extract_json(text: str) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not m:
            return None
        try:
            return json.loads(m.group(0))
        except Exception:
            return None


# ----------------------------
# Token/word localization helpers
# ----------------------------
_WORD_RE = re.compile(r"\S+")


def word_spans(sentence: str) -> List[Tuple[int, int, str]]:
    """Return list of (start_char, end_char, word_text) for non-whitespace tokens in the sentence."""
    spans: List[Tuple[int, int, str]] = []
    for m in _WORD_RE.finditer(sentence):
        spans.append((m.start(), m.end(), m.group(0)))
    return spans


def words_overlapping_char_range(
    w_spans: List[Tuple[int, int, str]],
    start: int,
    end: int,
) -> List[int]:
    """Return indices into w_spans that overlap [start, end)."""
    idxs: List[int] = []
    for i, (ws, we, _) in enumerate(w_spans):
        if max(ws, start) < min(we, end):
            idxs.append(i)
    return idxs


def summarize_top_spans(
    sentence: str,
    spans: Any,
    *,
    doc_word_base_1: int,
    top_k: int = 3,
) -> Tuple[str, str]:
    """Return (top_tokens_str, top_tokens_word_locs_str).

    - top_tokens_str: semicolon-separated top span texts
    - top_tokens_word_locs_str: semicolon-separated mapping token -> words (with 1-based doc word indices)

    Note: relies on model-provided `start`/`end` being character offsets within `sentence`.
    """
    if not isinstance(spans, list) or not spans:
        return "", ""

    # Keep only spans with numeric weight and plausible offsets
    cleaned = []
    for sp in spans:
        if not isinstance(sp, dict):
            continue
        try:
            w = float(sp.get("weight", 0.0))
        except Exception:
            continue
        try:
            s = int(sp.get("start"))
            e = int(sp.get("end"))
        except Exception:
            continue
        if s < 0 or e <= s or s > len(sentence) or e > len(sentence):
            continue
        txt = str(sp.get("text", "")).strip()
        if not txt:
            # fallback: take substring
            txt = sentence[s:e].strip()
        if not txt:
            continue
        cleaned.append((w, s, e, txt))

    if not cleaned:
        return "", ""

    # Sort by absolute weight (most significant) then take top_k
    cleaned.sort(key=lambda t: abs(t[0]), reverse=True)
    top = cleaned[:top_k]

    w_spans = word_spans(sentence)

    top_tokens = []
    top_locs = []
    for w, s, e, txt in top:
        top_tokens.append(txt)
        idxs = words_overlapping_char_range(w_spans, s, e)
        if idxs:
            # Map to 1-based word index within the document
            loc_parts = []
            for i in idxs:
                _, __, wtxt = w_spans[i]
                doc_word_idx_1 = doc_word_base_1 + i
                loc_parts.append(f"{wtxt}({doc_word_idx_1})")
            top_locs.append(f"{txt}→" + ",".join(loc_parts))
        else:
            top_locs.append(f"{txt}→(no word match)")

    return "; ".join(top_tokens), "; ".join(top_locs)


# ----------------------------
# Load input (TXT or CSV)
# ----------------------------
def load_texts(input_path: str, text_col: Optional[str]) -> List[str]:
    if input_path.lower().endswith(".txt"):
        with open(input_path, "r", encoding="utf-8") as f:
            text = f.read()
        # Strip Project Gutenberg / encoding artifacts
        text = strip_gutenberg_artifacts(text)
        return [text]

    if input_path.lower().endswith(".csv"):
        if not text_col:
            raise ValueError("For CSV input, --text_col is required.")
        df = pd.read_csv(input_path)
        if text_col not in df.columns:
            raise ValueError(f"Column '{text_col}' not found.")
        series = df[text_col].fillna("").astype(str)
        # Strip Project Gutenberg / encoding artifacts
        series = series.apply(strip_gutenberg_artifacts)
        return series.tolist()

    raise ValueError("Input must be .txt or .csv")


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="crime_and_punishment.txt")
    ap.add_argument("--text_col", default=None)
    ap.add_argument("--output_csv", default="flagged_sentences.csv")
    ap.add_argument("--model", default="llama3")
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--splitter", choices=["regex", "spacy"], default="regex",
                    help="Sentence splitter to use. 'regex' has no extra deps; 'spacy' requires spaCy installed.")
    ap.add_argument("--spacy_model", default="en_core_web_sm",
                    help="spaCy model name (only used if --splitter spacy).")
    ap.add_argument("--min_chars", type=int, default=20)
    ap.add_argument("--max_sent_chars", type=int, default=400)
    ap.add_argument("--z_thresh", type=float, default=2.0)
    ap.add_argument("--top_pos", type=int, default=0,
                    help="If >0, ignore --z_thresh and output the top N positive robust_z outliers.")
    ap.add_argument("--top_neg", type=int, default=0,
                    help="If >0, ignore --z_thresh and output the top N negative robust_z outliers.")
    ap.add_argument("--top_metric", choices=["robust_z", "intensity"], default="robust_z",
                    help="Metric to use for --top_pos/--top_neg selection. 'robust_z' uses MAD z-scores; 'intensity' uses raw intensity.")
    ap.add_argument("--score_mode", choices=["intensity", "signed"], default="intensity",
                    help="How to rank outliers. 'intensity' uses raw intensity only. 'signed' uses intensity*valence to separate extreme positive vs extreme negative.")
    ap.add_argument("--sleep", type=float, default=0.0)
    ap.add_argument("--show_progress", action="store_true",
                    help="Print progress updates while scoring sentences.")
    ap.add_argument("--progress_every", type=int, default=50,
                    help="Print progress every N sentences (only if --show_progress).")
    args = ap.parse_args()

    texts = load_texts(args.input, args.text_col)

    def get_sentences(text: str) -> List[str]:
        if args.splitter == "regex":
            return split_sentences_regex(text)
        # spaCy mode: import lazily so the script runs even if spaCy isn't installed
        try:
            import spacy  # type: ignore
        except Exception as e:
            raise SystemExit(
                "spaCy is not available in this environment. "
                "Either install it (and its deps) or run with --splitter regex.\n" + str(e)
            )
        try:
            nlp = spacy.load(args.spacy_model)
        except Exception as e:
            raise SystemExit(
                f"Could not load spaCy model '{args.spacy_model}'. "
                f"Try: python -m spacy download {args.spacy_model}\n" + str(e)
            )
        doc = nlp(text)
        return [s.text.strip() for s in doc.sents if s.text and s.text.strip()]

    records = []
    sent_idx = 0

    total_sentences = 0
    docs_sentences: List[List[str]] = []

    # Pre-split so we can show deterministic progress totals.
    for text in texts:
        sents = get_sentences(text)
        docs_sentences.append(sents)
        total_sentences += len(sents)

    processed = 0

    # Track 1-based word positions within each source document.
    # For .txt input there is typically one doc; for CSV each row becomes a doc.
    doc_word_offsets_1 = [1 for _ in range(len(docs_sentences))]

    for doc_idx, sents in enumerate(docs_sentences):
        for s in sents:
            processed += 1
            if args.show_progress and (processed == 1 or processed % max(1, args.progress_every) == 0 or processed == total_sentences):
                pct = (processed / total_sentences) * 100 if total_sentences else 100.0
                print(f"Progress: {processed}/{total_sentences} sentences ({pct:.1f}%)", end="\r", flush=True)

            # Compute word spans/length early so we can track word indices even for skipped sentences.
            # (This keeps word positions stable relative to the processed text.)
            w_spans_for_count = word_spans(s)
            sent_word_count = len(w_spans_for_count)

            if len(s) < args.min_chars:
                # Advance the document-level word counter even if we skip scoring
                doc_word_offsets_1[doc_idx] += sent_word_count
                continue
            if len(s) > args.max_sent_chars:
                s = s[:args.max_sent_chars]
                # Recompute after truncation
                w_spans_for_count = word_spans(s)
                sent_word_count = len(w_spans_for_count)

            raw = ollama_generate(
                build_prompt(s),
                model=args.model,
                host=args.host,
            )

            obj = extract_json(raw)
            if obj is None:
                # Still advance word counter so document word positions remain stable
                doc_word_offsets_1[doc_idx] += sent_word_count
                continue

            intensity = float(obj.get("intensity", 0.0))
            # valence in [-1, +1]; default to 0 if missing
            try:
                valence = float(obj.get("valence", 0.0))
            except Exception:
                valence = 0.0
            # Clamp to [-1, 1] to keep the downstream score well-behaved
            valence = max(-1.0, min(1.0, valence))

            spans = obj.get("spans", [])
            notes = obj.get("notes", "")

            top_tokens, top_token_word_locs = summarize_top_spans(
                s,
                spans,
                doc_word_base_1=doc_word_offsets_1[doc_idx],
                top_k=3,
            )

            records.append({
                "doc_index": doc_idx,
                "sentence_index": sent_idx,
                "sentence": s,
                "intensity": intensity,
                "valence": valence,
                "signed_intensity": intensity * valence,
                "spans_json": json.dumps(spans, ensure_ascii=False),
                "notes": notes,
                "top3_tokens": top_tokens,
                "top3_token_word_locs": top_token_word_locs,
            })
            sent_idx += 1

            # Advance the document-level word counter after scoring this sentence
            doc_word_offsets_1[doc_idx] += sent_word_count

            if args.sleep > 0:
                time.sleep(args.sleep)

    if args.show_progress:
        # ensure the progress line ends cleanly
        print()

    df = pd.DataFrame(records)

    if df.empty:
        # Still write an empty CSV with the expected columns
        empty_cols = ["doc_index", "sentence_index", "sentence", "intensity", "valence", "signed_intensity", "spans_json", "notes", "robust_z", "top3_tokens", "top3_token_word_locs"]
        pd.DataFrame(columns=empty_cols).to_csv(args.output_csv, index=False)
        print(f"Wrote 0 flagged sentences → {args.output_csv}")
        print("\nTop examples:")
        print("(none)")
        return

    scores = df["intensity"].to_numpy()
    df["robust_z"] = robust_z_mad(scores)

    # Flagging modes:
    # - Default: z-threshold on |robust_z|
    # - If --top_pos/--top_neg are set: take top-N positive and/or negative robust_z outliers
    if (args.top_pos and args.top_pos > 0) or (args.top_neg and args.top_neg > 0):
        pos_n = max(0, int(args.top_pos))
        neg_n = max(0, int(args.top_neg))

        # Ranking strategy: intensity-only (existing) or signed intensity (positive vs negative)
        metric = args.top_metric
        use_signed = (args.score_mode == "signed")

        # If using signed scoring, drop sentences that are effectively neutral direction.
        # This aligns with the use-case: only extreme positive vs extreme negative.
        base_df = df
        if use_signed:
            base_df = df[np.abs(df["signed_intensity"]) > 1e-9].copy()

        if use_signed:
            # Positive = most positive signed_intensity, Negative = most negative signed_intensity
            pos = base_df.sort_values("signed_intensity", ascending=False).head(pos_n).copy() if pos_n > 0 else base_df.head(0).copy()
            neg = base_df.sort_values("signed_intensity", ascending=True).head(neg_n).copy() if neg_n > 0 else base_df.head(0).copy()
        elif metric == "intensity":
            # Positive = highest intensity, Negative = lowest intensity (note: this is intensity, not sentiment)
            pos = base_df.sort_values("intensity", ascending=False).head(pos_n).copy() if pos_n > 0 else base_df.head(0).copy()
            neg = base_df.sort_values("intensity", ascending=True).head(neg_n).copy() if neg_n > 0 else base_df.head(0).copy()
        else:
            # Positive = highest robust_z, Negative = lowest robust_z
            pos = base_df.sort_values("robust_z", ascending=False).head(pos_n).copy() if pos_n > 0 else base_df.head(0).copy()
            neg = base_df.sort_values("robust_z", ascending=True).head(neg_n).copy() if neg_n > 0 else base_df.head(0).copy()

        pos["outlier_side"] = "positive"
        neg["outlier_side"] = "negative"

        # Order output: positives first, then negatives.
        # Within each side, order by the chosen metric with the most extreme first.
        sort_col = "signed_intensity" if use_signed else ("intensity" if args.top_metric == "intensity" else "robust_z")

        pos_sorted = pos.sort_values(sort_col, ascending=False).copy() if not pos.empty else pos
        neg_sorted = neg.sort_values(sort_col, ascending=True).copy() if not neg.empty else neg

        flagged = pd.concat([pos_sorted, neg_sorted], ignore_index=True)

        flagged.to_csv(args.output_csv, index=False)
        print(f"Wrote {len(flagged)} flagged sentences → {args.output_csv} (top_pos={pos_n}, top_neg={neg_n}, metric={args.top_metric})")

        print("\nTop examples:")
        preview_cols = ["outlier_side", "robust_z", "intensity", "valence", "signed_intensity", "top3_tokens", "top3_token_word_locs", "sentence"]
        print(flagged[preview_cols].head(10).to_string(index=False))
    else:
        flagged = df[np.abs(df["robust_z"]) >= args.z_thresh].copy()
        flagged.sort_values("robust_z", ascending=False, inplace=True)

        flagged.to_csv(args.output_csv, index=False)
        print(f"Wrote {len(flagged)} flagged sentences → {args.output_csv}")

        print("\nTop examples:")
        print(flagged[["robust_z", "intensity", "valence", "signed_intensity", "top3_tokens", "top3_token_word_locs", "sentence"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()