import argparse
import json
from pathlib import Path


VARIANT_ORDER = {"BASE": 0, "EMBER-PROMPT": 1, "EMBER-AGENT": 2, "SFT": 3, "SFT+RL": 4}


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def sort_key(row):
    meta = row.get("meta") or {}
    return (
        VARIANT_ORDER.get(meta.get("variant_label"), 999),
        int(meta.get("topic_index_1based") or 0),
        int(meta.get("round") or 0),
        meta.get("record_id") or "",
    )


def main():
    parser = argparse.ArgumentParser(description="Merge generated EMBER JSONL files by meta.record_id.")
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rows_by_id = {}
    for input_path in args.inputs:
        for row in read_jsonl(input_path) or []:
            record_id = (row.get("meta") or {}).get("record_id")
            if not record_id:
                continue
            rows_by_id[record_id] = row

    rows = sorted(rows_by_id.values(), key=sort_key)
    write_jsonl(args.output, rows)
    print(json.dumps({"output": args.output, "rows": len(rows), "inputs": args.inputs}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
