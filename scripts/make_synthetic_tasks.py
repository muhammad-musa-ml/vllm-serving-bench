"""Generate the synthetic quality tasks (seeded, so the output is reproducible byte for byte).

120 items: 40 arithmetic with an exact integer answer, 40 extractions of one fact from a made-up
order note, 40 format-following items (14 exact-text, 13 JSON-object, 13 line-count) checked by
`servebench.quality.score_task`. Everything here is synthetic.

Run:  python scripts/make_synthetic_tasks.py [--out quality/synthetic_tasks.jsonl] [--seed 20261024]
"""
import argparse
import json
import random
import string
from pathlib import Path

CITIES = ["Leeds", "Porto", "Gdansk", "Kobe", "Lyon", "Perth", "Quito", "Graz", "Malmo", "Tartu",
          "Bergen", "Cork", "Split", "Turin", "Ghent", "Salta"]
CARRIERS = ["Harbor Freight Co", "Northline", "Blue Mile", "Kestrel Cargo", "Tallow Express",
            "Ridgeway", "Orchard Post", "Lantern Logistics"]
NAMES = ["Ana", "Bram", "Chen", "Dara", "Eli", "Femi", "Gus", "Hana", "Ivo", "Juno", "Kofi",
         "Lena", "Milo", "Nia", "Omar", "Pia"]
N_ARITH, N_EXTRACT, N_EXACT, N_JSON, N_LINES = 40, 40, 14, 13, 13


def arith(rng):
    op = rng.choice(["+", "-", "*"])
    if op == "*":
        a, b = rng.randint(12, 99), rng.randint(12, 99)
        value = a * b
    else:
        a, b = rng.randint(100, 999), rng.randint(100, 999)
        value = a + b if op == "+" else a - b
    return {"kind": "arith", "scorer": "int",
            "prompt": f"What is {a} {op} {b}? Reply with the number only.", "expected": value}


def extract(rng):
    number = rng.randint(10000, 99999)
    city, carrier = rng.choice(CITIES), rng.choice(CARRIERS)
    packed = f"2026-{rng.randint(1, 6):02d}-{rng.randint(1, 28):02d}"
    shipped = f"2026-{rng.randint(7, 12):02d}-{rng.randint(1, 28):02d}"
    note = (f"Order #{number} was packed on {packed} and shipped on {shipped} to {city} "
            f"by {carrier}.")
    question, expected = rng.choice([
        ("What is the order number?", str(number)),
        ("Which city was it shipped to?", city),
        ("Which carrier shipped it?", carrier),
        ("On what date was it shipped?", shipped),
        ("On what date was it packed?", packed),
    ])
    return {"kind": "extract", "scorer": "text",
            "prompt": f"{note}\n\n{question} Reply with that value only and nothing else.",
            "expected": expected}


def fmt_exact(rng):
    code = "".join(rng.choice(string.ascii_uppercase) for _ in range(6))
    return {"kind": "format-exact", "scorer": "exact",
            "prompt": f"Reply with exactly this text and nothing else: {code}", "expected": code}


def fmt_json(rng):
    name, count = rng.choice(NAMES), rng.randint(2, 90)
    return {"kind": "format-json", "scorer": "json",
            "prompt": (f'Return only a JSON object with the key "name" set to the string "{name}" '
                       f'and the key "count" set to the integer {count}. Output nothing else.'),
            "expected": {"name": name, "count": count}}


def fmt_lines(rng):
    n = rng.randint(3, 6)
    return {"kind": "format-lines", "scorer": "lines",
            "prompt": (f'Write exactly {n} lines. Start every line with "- " and put a different '
                       "fruit name after it. Output nothing else."),
            "expected": {"n": n, "prefix": "- "}}


def build(seed):
    rng = random.Random(seed)
    items = ([arith(rng) for _ in range(N_ARITH)] + [extract(rng) for _ in range(N_EXTRACT)]
             + [fmt_exact(rng) for _ in range(N_EXACT)] + [fmt_json(rng) for _ in range(N_JSON)]
             + [fmt_lines(rng) for _ in range(N_LINES)])
    rng.shuffle(items)
    for i, item in enumerate(items):
        item["id"] = f"syn-{i:03d}"
        item["set"] = "synthetic-tasks"
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="quality/synthetic_tasks.jsonl")
    ap.add_argument("--seed", type=int, default=20261024)
    args = ap.parse_args()
    items = build(args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
    print(f"wrote {len(items)} items to {out}")


if __name__ == "__main__":
    main()
