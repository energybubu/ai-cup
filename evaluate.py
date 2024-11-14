"""Evaluate the model."""

import argparse
import json


def parse_arguments():
    """Parse the arguments from the command line."""
    parser = argparse.ArgumentParser(description="Evaluate the model.")
    parser.add_argument(
        "--question_path",
        type=str,
        default="datasets/dataset/preliminary/questions_example.json",
        help="讀取發布題目路徑",
    )
    parser.add_argument(
        "--pred_path",
        type=str,
        required=True,
        help="The path to the prediction file.",
    )
    parser.add_argument(
        "--gt_path",
        type=str,
        default="datasets/dataset/preliminary/ground_truths_example.json",
        help="The path to the ground truth file.",
    )

    return parser.parse_args()


def evaluate(args):
    """Evaluate the model."""
    with open(args.question_path) as f:
        qs_ref = json.load(f)["questions"]
        sources = {q["qid"]: q["source"] for q in qs_ref}

    with open(args.pred_path) as f:
        pred_data = json.load(f)["answers"]
        pred = {p["qid"]: p["retrieve"] for p in pred_data}

    with open(args.gt_path) as f:
        gt_data = json.load(f)["ground_truths"]
        gt = {g["qid"]: g["retrieve"] for g in gt_data}
        q_cat = {g["qid"]: g["category"] for g in gt_data}

    assert set(pred.keys()) == set(
        gt.keys()
    ), "Prediction and ground truth have different qids."
    assert (
        len(pred) == len(gt) == 150
    ), "Prediction and ground truth have wrong number of qids."

    correct_cnt = {"all": [], "insurance": [], "finance": [], "faq": []}
    for qid in pred.keys():
        assert (
            pred[qid] in sources[qid]
        ), f"Prediction with qid {qid} does not exist in the sources."
        correct_cnt["all"].append(pred[qid] == gt[qid])
        correct_cnt[q_cat[qid]].append(pred[qid] == gt[qid])

    print(f"Total Accuracy: {sum(correct_cnt["all"])/len(correct_cnt["all"]):.2f}")
    print(
        f"Insurance Accuracy: {sum(correct_cnt["insurance"])/len(correct_cnt["insurance"]):.2f}"
    )
    print(
        f"Finance Accuracy: {sum(correct_cnt["finance"])/len(correct_cnt["finance"]):.2f}"
    )
    print(f"FAQ Accuracy: {sum(correct_cnt["faq"])/len(correct_cnt["faq"]):.2f}")


if __name__ == "__main__":
    args = parse_arguments()
    evaluate(args)
