"""Main function of the project."""

import argparse

import data.io as io
from models.bm25_retrieve import bm25_rerank
from models.cohere_retrieve import cohere_rerank
from models.conan import conan_rerank
from models.qwen import qwen_rerank
from models.zpoint import zpoint_rerank
from models.hybrid import hybrid_rerank
from models.qwen_gpt import qwen_gpt_rerank


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
    parser.add_argument(
        "--use_cache",
        action=argparse.BooleanOptionalAction,
        help="use cached preprocessed documents"
    )

    return parser.parse_args()  # 解析參數


def get_rerank_model(model_name: str):
    """Get the model function based on the model name."""
    if model_name == "bm25":
        return bm25_rerank
    elif model_name == "cohere":
        return cohere_rerank
    elif model_name == "conan":
        return conan_rerank
    elif model_name == "qwen":
        return qwen_rerank
    elif model_name == "zpoint":
        return zpoint_rerank
    elif model_name == "hybrid":
        return hybrid_rerank
    elif model_name == "qwen_gpt":
        return qwen_gpt_rerank

    raise ValueError(f"Model {model_name} is not supported.")


def exp(exp_args: argparse.Namespace):
    """Main Experiment Function."""
    qs_ref, corpus_dict_insurance, corpus_dict_finance, key_to_source_dict = (
        io.parse_input(exp_args)
    )
    rerank_model = get_rerank_model(exp_args.model)

    answer_dict = rerank_model(
        qs_ref, corpus_dict_insurance, corpus_dict_finance, key_to_source_dict
    )
    io.write_answer_to_json(exp_args, answer_dict)


if __name__ == "__main__":
    args = parse_arguments()
    exp(args)
