"""Main function of the project."""

import argparse
import main_io
from models.bm25_retrieve import bm25_rerank
from models.reranker import reranker
from models.qwen import qwen


def exp(exp_args: argparse.Namespace):
    """Main Experiment Function."""
    qs_ref, corpus_dict_insurance, corpus_dict_finance, key_to_source_dict = (
        main_io.parse_input(exp_args)
    )
    if exp_args.model == "bm25":
        model = bm25_rerank
    elif exp_args.model == "reranker":
        model = reranker
    elif exp_args.model == "qwen":
        model = qwen
    else:
        raise ValueError("Model not supported.")

    answer_dict = model(
        qs_ref,
        corpus_dict_insurance,
        corpus_dict_finance,
        key_to_source_dict,
    )
    main_io.write_answer_to_json(exp_args, answer_dict)


if __name__ == "__main__":
    # 使用argparse解析命令列參數
    parser = argparse.ArgumentParser(description="Process some paths and files.")
    parser.add_argument(
        "--question_path", type=str, required=True, help="讀取發布題目路徑"
    )  # 問題文件的路徑
    parser.add_argument(
        "--source_path", type=str, required=True, help="讀取參考資料路徑"
    )  # 參考資料的路徑
    parser.add_argument(
        "--output_path", type=str, required=True, help="輸出符合參賽格式的答案路徑"
    )  # 答案輸出的路徑

    args = parser.parse_args()  # 解析參數
    exp(args)
