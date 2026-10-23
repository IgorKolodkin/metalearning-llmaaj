"""Convert train/test CSV to model input JSONL, removing test gold and translations."""
import argparse
import json
from pathlib import Path

from common import load_dataset, write_jsonl


def model_rows(rows, train=False, language="original"):
    if language not in {"original", "es", "en"}:
        raise ValueError("Language must be original, es or en")
    result = []
    for row in rows:
        content = row
        if language != "original":
            field = f"input_{language}"
            if field not in row:
                raise ValueError(f"Dataset has no {field}")
            content = json.loads(row[field]) if isinstance(row[field], str) else row[field]
        value = {"item_id": row["item_id"], "topic": row["topic"],
                 "context": content["context"], "conversation": content["conversation"]}
        if train:
            value["label"] = row["label"]
        result.append(value)
    return result


def prepare(train, test, out_dir, language="original"):
    train_rows = load_dataset(train, require_labels=True)
    test_rows = load_dataset(test)
    if {row["item_id"] for row in train_rows} & {row["item_id"] for row in test_rows}:
        raise ValueError("Train and test IDs overlap")
    train_values = model_rows(train_rows, train=True, language=language)
    test_values = model_rows(test_rows, language=language)
    output = Path(out_dir)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use an empty output directory to avoid stale inputs")
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "train.jsonl", train_values)
    write_jsonl(output / "test.jsonl", test_values)
    return {"train": len(train_values), "test": len(test_values), "language": language,
            "train_path": str((output / "train.jsonl").resolve()),
            "test_path": str((output / "test.jsonl").resolve()), "test_labels_removed": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--language", choices=["original", "es", "en"], default="original")
    args = parser.parse_args()
    try:
        result = prepare(args.train, args.test, args.out_dir, args.language)
    except (ValueError, OSError, KeyError) as error:
        parser.exit(2, f"Input preparation failed: {error}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
