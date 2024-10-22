import argparse
import json

def parse_arguments():
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
        required=True,
        help="The path to the ground truth file.",
    )

    return parser.parse_args() 

def evaluate(args):
    with open(args.question_path) as f:
        qs_ref = json.load(f)
        
    with open(args.pred_path) as f:
        pred = json.load(f)
        
    with open(args.gt_path) as f:
        gt = json.load(f)

    assert len(pred) == len(gt)
    correct = 0
    for pred_item, gt_item in zip(pred, gt):
        if pred_item["qid"] == gt_item["qid"] and pred_item["retrieve"] == gt_item["retrieve"]:
            correct += 1

if __name__ == "__main__":
    args = parse_arguments()
    evaluate(args)