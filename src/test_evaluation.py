"""Contract tests: label-only scoring, weighting, strict IDs and label removal."""
import copy
import csv
import json
from pathlib import Path
import tempfile
import unittest

from common import TOPICS, load_dataset, load_labels, read_records, write_jsonl
from evaluate import evaluate, score
from prepare_inputs import model_rows, prepare


def fixture():
    return [{"item_id": f"test-{i}", "topic": topic, "context": {"balance": 10},
             "conversation": [{"role": "assistant", "content": 'Text, with "quotes"\nand a newline'}],
             "label": "good" if i % 2 == 0 else "bad"}
            for i, topic in enumerate(TOPICS)]


def predictions(rows):
    return [{"item_id": row["item_id"], "label": row["label"]} for row in rows]


class EvaluationTests(unittest.TestCase):
    def test_perfect_wrong_and_order_independent(self):
        rows = fixture()
        self.assertEqual(score(rows, list(reversed(predictions(rows))))["score"], 1)
        wrong = [{"item_id": row["item_id"], "label": "bad" if row["label"] == "good" else "good"} for row in rows]
        self.assertEqual(score(rows, wrong)["score"], 0)

    def test_equal_topic_weights(self):
        rows = fixture()
        rows += [dict(rows[0], item_id=f"extra-{i}") for i in range(6)]
        guesses = predictions(rows)
        for row, guess in zip(rows, guesses):
            if row["topic"] == TOPICS[0]:
                guess["label"] = "bad"
        result = score(rows, guesses)
        self.assertEqual(result["macro_topic_accuracy"], .75)
        self.assertAlmostEqual(result["accuracy"], .3)
        self.assertEqual(result["by_topic"][TOPICS[0]]["confusion_matrix"], [[0, 7], [0, 0]])

    def test_invalid_predictions_rejected(self):
        rows = fixture()
        values = predictions(rows)
        variants = [values[:-1], values + values[:1], [dict(values[0], item_id="unknown")] + values[1:],
                    [dict(values[0], label="neutral")] + values[1:],
                    [dict(values[0], probs={"good": 1, "bad": 0})] + values[1:]]
        for variant in variants:
            with self.subTest(variant=variant), self.assertRaises(ValueError):
                score(rows, variant)

    def test_private_gold_and_conflicting_gold(self):
        rows = fixture()
        hidden = [{key: value for key, value in row.items() if key != "label"} for row in rows]
        self.assertEqual(score(hidden, predictions(rows), predictions(rows))["score"], 1)
        with self.assertRaises(ValueError):
            score(hidden, predictions(rows))
        conflict = predictions(rows)
        conflict[0]["label"] = "bad"
        with self.assertRaises(ValueError):
            score(rows, predictions(rows), conflict)

    def test_all_topics_required(self):
        rows = fixture()[:-1]
        with self.assertRaises(ValueError):
            score(rows, predictions(rows))

    def test_predict_input_contains_no_gold_or_extra_fields(self):
        rows = fixture()
        for row in rows:
            row["input_en"] = json.dumps({"context": row["context"], "conversation": row["conversation"]})
        converted = model_rows(rows, language="en")
        self.assertTrue(all(set(row) == {"item_id", "topic", "context", "conversation"} for row in converted))
        self.assertTrue(all("label" in row for row in model_rows(rows, train=True)))

    def test_csv_quoting_and_label_only_jsonl(self):
        rows = fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            test = root / "test.csv"
            with test.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows({key: json.dumps(value) if key in {"context", "conversation"} else value
                                  for key, value in row.items()} for row in rows)
            guesses = root / "predictions.jsonl"
            write_jsonl(guesses, predictions(rows))
            self.assertEqual(evaluate(test, guesses)["score"], 1)
            self.assertEqual(load_dataset(test)[0]["conversation"], rows[0]["conversation"])

    def test_duplicate_headers_and_wrong_row_width(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.csv"
            for content in ["item_id,label,label\nx,good,good\n", "item_id,label\nx,good,extra\n", "item_id,label\nx\n"]:
                path.write_text(content)
                with self.subTest(content=content), self.assertRaises(ValueError):
                    load_labels(path)

    def test_prepare_train_overlap_and_stale_directory_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train, test = root / "train.jsonl", root / "test.jsonl"
            write_jsonl(train, fixture())
            write_jsonl(test, fixture())
            with self.assertRaises(ValueError):
                prepare(train, test, root / "input")
            rows = copy.deepcopy(fixture())
            for row in rows:
                row["item_id"] += "-held-out"
            write_jsonl(test, rows)
            prepare(train, test, root / "input")
            self.assertTrue(all("label" not in row for row in read_records(root / "input/test.jsonl")))
            with self.assertRaises(ValueError):
                prepare(train, test, root / "input")


if __name__ == "__main__":
    unittest.main()
