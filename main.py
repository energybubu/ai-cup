"""Main function of the project."""

import argparse

import data.io as io
from models.bm25_retrieve import bm25_rerank
from models.cohere_retrieve import cohere_rerank
from models.conan import conan_rerank
from models.qwen import qwen_rerank
from models.zhinao import zhinao_rerank


def parse_arguments():
    """Parse the arguments from the command line."""
    parser = argparse.ArgumentParser(description="Process some paths and files.")
    parser.add_argument(
        "--question_path",
        type=str,
        default="datasets/dataset/preliminary/questions_example.json",
        help="讀取發布題目路徑",
    )  # 問題文件的路徑
    parser.add_argument(
        "--source_path",
        type=str,
        default="datasets/reference",
        help="讀取參考資料路徑",
    )  # 參考資料的路徑
    parser.add_argument(
        "--output_path",
        type=str,
        default="datasets/dataset/preliminary/pred_retrieve.json",
        help="輸出符合參賽格式的答案路徑",
    )  # 答案輸出的路徑
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        default="bm25",
        help="選擇模型",
    )

    return parser.parse_args()  # 解析參數


def exp(exp_args: argparse.Namespace):
    """Main Experiment Function."""
    qs_ref, corpus_dict_insurance, corpus_dict_finance, key_to_source_dict = (
        io.parse_input(exp_args)
    )
    if exp_args.model == "bm25":
        model = bm25_rerank
    elif exp_args.model == "cohere":
        model = cohere_rerank
    elif exp_args.model == "qwen":
        model = qwen_rerank
    elif exp_args.model == "conan":
        model = conan_rerank
    elif exp_args.model == "zhinao":
        model = zhinao_rerank
    else:
        raise ValueError("Model not supported.")

    answer_dict = model(
        qs_ref, corpus_dict_insurance, corpus_dict_finance, key_to_source_dict
    )
    io.write_answer_to_json(exp_args, answer_dict)


if __name__ == "__main__":
    args = parse_arguments()
    exp(args)
