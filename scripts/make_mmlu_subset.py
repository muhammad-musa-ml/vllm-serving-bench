"""Draw the fixed MMLU question subset used as the quality set.

Source: the `cais/mmlu` dataset on Hugging Face (MIT licence), `all` configuration, test split,
pinned to one dataset revision. The script downloads the parquet file, takes 5 questions from each
subject with a fixed seed, and writes `quality/mmlu_subset.jsonl` plus
`quality/mmlu_subset.source.json` (URL, date fetched, SHA-256 of the download, seed).

Run:  python -I scripts/make_mmlu_subset.py [--out quality] [--seed 20261024] [--per-subject 5]
"""
import argparse
import datetime
import hashlib
import json
import random
import sys
import tempfile
import urllib.request
from pathlib import Path

import pyarrow.parquet as pq

DATASET_REVISION = "c30699e8356da336a370243923dbaf21066bb9fe"
URL = (f"https://huggingface.co/datasets/cais/mmlu/resolve/{DATASET_REVISION}"
       "/all/test-00000-of-00001.parquet")
LETTERS = "ABCD"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="quality")
    ap.add_argument("--seed", type=int, default=20261024)
    ap.add_argument("--per-subject", type=int, default=5)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "mmlu-test.parquet"
        with urllib.request.urlopen(URL, timeout=120) as resp, open(target, "wb") as f:
            f.write(resp.read())
        fetched = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        table = pq.read_table(target).to_pylist()

    by_subject: dict[str, list[dict]] = {}
    for row in table:
        by_subject.setdefault(row["subject"], []).append(row)

    rng = random.Random(args.seed)
    chosen = []
    for subject in sorted(by_subject):
        rows = by_subject[subject]
        if len(rows) < args.per_subject:
            sys.exit(f"{subject} has only {len(rows)} questions")
        for row in rng.sample(rows, args.per_subject):
            chosen.append({
                "id": f"mmlu-{len(chosen):04d}",
                "set": "mmlu-subset",
                "subject": subject,
                "question": row["question"],
                "choices": list(row["choices"]),
                "answer": LETTERS[int(row["answer"])],
                "scorer": "mcq",
            })

    with open(out / "mmlu_subset.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for item in chosen:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    source = {
        "dataset": "cais/mmlu", "config": "all", "split": "test", "licence": "MIT",
        "dataset_revision": DATASET_REVISION, "url": URL, "fetched_utc": fetched,
        "download_sha256": digest, "seed": args.seed, "per_subject": args.per_subject,
        "n_subjects": len(by_subject), "n_questions": len(chosen),
        "n_rows_in_split": len(table),
    }
    (out / "mmlu_subset.source.json").write_text(json.dumps(source, indent=2) + "\n",
                                                 encoding="utf-8", newline="\n")
    print(json.dumps(source, indent=2))


if __name__ == "__main__":
    main()
