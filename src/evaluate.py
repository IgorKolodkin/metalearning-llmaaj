"""Evaluate good/bad labels against a CSV or JSONL dataset. No model/API calls."""
import argparse
import json
from pathlib import Path

from common import LABELS, TOPICS, indexed, load_dataset, load_labels


def metrics(gold, predicted):
    if not gold or len(gold) != len(predicted):
        raise ValueError("Nonempty, aligned labels are required")
    matrix = [[0, 0], [0, 0]]
    for actual, guess in zip(gold, predicted):
        if actual not in LABELS or guess not in LABELS:
            raise ValueError("Labels must be exactly good or bad")
        matrix[LABELS.index(actual)][LABELS.index(guess)] += 1
    count = len(gold)
    correct = matrix[0][0] + matrix[1][1]
    accuracy = correct / count
    expected = sum(sum(matrix[i]) * sum(matrix[j][i] for j in range(2)) for i in range(2)) / count**2
    kappa = (accuracy - expected) / (1 - expected) if expected < 1 else 0.0
    return {"n": count, "correct": correct, "accuracy": accuracy, "kappa": kappa,
            "confusion_matrix": matrix}


def score(test_rows, prediction_rows, label_rows=None):
    test = indexed(test_rows)
    predictions = indexed(prediction_rows)
    labels = indexed(label_rows) if label_rows is not None else test
    if set(predictions) != set(test) or set(labels) != set(test):
        raise ValueError("Prediction and gold IDs must match the test set exactly")
    groups = {topic: ([], []) for topic in TOPICS}
    gold, predicted = [], []
    for iid, row in test.items():
        topic = row.get("topic")
        if topic not in TOPICS:
            raise ValueError("Unknown or missing topic")
        actual = labels[iid].get("label")
        if "label" in row and row["label"] != actual:
            raise ValueError("External gold conflicts with labels in the test dataset")
        prediction = predictions[iid]
        if set(prediction) != {"item_id", "label"}:
            raise ValueError("Predictions require exactly item_id and label; probabilities are not used")
        guess = prediction["label"]
        if actual not in LABELS or guess not in LABELS:
            raise ValueError("Labels must be exactly good or bad")
        gold.append(actual)
        predicted.append(guess)
        groups[topic][0].append(actual)
        groups[topic][1].append(guess)
    if any(not actual for actual, _ in groups.values()):
        raise ValueError("Evaluation requires all four topics")
    by_topic = {topic: metrics(*values) for topic, values in groups.items()}
    result = metrics(gold, predicted)
    result.update(primary_metric="macro_topic_accuracy",
                  score=sum(value["accuracy"] for value in by_topic.values()) / len(TOPICS),
                  macro_topic_accuracy=sum(value["accuracy"] for value in by_topic.values()) / len(TOPICS),
                  macro_topic_kappa=sum(value["kappa"] for value in by_topic.values()) / len(TOPICS),
                  confusion_matrix_order=list(LABELS), by_topic=by_topic)
    return result


def evaluate(test, predictions, labels=None):
    return score(load_dataset(test), load_labels(predictions), load_labels(labels) if labels else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", required=True, help="Labeled public test or unlabeled private test")
    parser.add_argument("--predictions", required=True, help="CSV/JSONL containing exactly item_id,label")
    parser.add_argument("--labels", help="Separate organizer gold CSV/JSONL for an unlabeled test")
    parser.add_argument("--out", help="Save the report as JSON")
    args = parser.parse_args()
    try:
        result = evaluate(args.test, args.predictions, args.labels)
    except (ValueError, OSError, KeyError) as error:
        parser.exit(2, f"Evaluation failed: {error}\n")
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.out:
        path = Path(args.out)
        inputs = [Path(value).resolve() for value in (args.test, args.predictions, args.labels) if value]
        if path.resolve() in inputs:
            parser.exit(2, "--out must not overwrite an input file\n")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
