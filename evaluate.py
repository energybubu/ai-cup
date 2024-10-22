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
        qs_ref = json.load(f)['questions']
        sources = {
            q['qid']: q['source'] for q in qs_ref
        }

    with open(args.pred_path) as f:
        pred = json.load(f)['answers']
        pred = {
            p['qid']: p['retrieve'] for p in pred
        }
        
    with open(args.gt_path) as f:
        gt = json.load(f)['ground_truths']
        gt = {
            g['qid']: g['retrieve'] for g in gt
        }

    assert set(pred.keys()) == set(gt.keys()), "Prediction and ground truth have different qids."
    assert len(pred) == len(gt) == 150, "Prediction and ground truth have wrong number of qids."

    correct = 0
    for qid in pred.keys():
        assert pred[qid] in sources[qid], f"Prediction with qid {qid} does not exist in the sources."
        if pred[qid] == gt[qid]:
            correct += 1

    print(f"Accuracy: {correct/len(pred):.2f}")

    
    

if __name__ == "__main__":
    args = parse_arguments()
    evaluate(args)