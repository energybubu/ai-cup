import time

import cohere
from tqdm import tqdm

from models.constant import *


def cohere_retrieve(qs, source, corpus_dict):
    co = cohere.ClientV2(cohere_api_key)

    documents = []
    for file_num in source:
        documents.append(corpus_dict[file_num])

    response = co.rerank(
        model="rerank-multilingual-v3.0", query=qs, documents=documents, top_n=1
    )

    time.sleep(cohere_api_wait_time)

    return source[response.results[0].index]


def cohere_rerank(
    qs_ref,
    corpus_dict_insurance,
    corpus_dict_finance,
    key_to_source_dict,
):
    answer_dict = {"answers": []}  # 初始化字典

    for q_dict in tqdm(qs_ref["questions"]):
        if q_dict["category"] == "finance":
            # 進行檢索
            retrieved = cohere_retrieve(
                q_dict["query"], q_dict["source"], corpus_dict_finance
            )
            # 將結果加入字典
            answer_dict["answers"].append({"qid": q_dict["qid"], "retrieve": retrieved})

        elif q_dict["category"] == "insurance":
            retrieved = cohere_retrieve(
                q_dict["query"], q_dict["source"], corpus_dict_insurance
            )
            answer_dict["answers"].append({"qid": q_dict["qid"], "retrieve": retrieved})

        elif q_dict["category"] == "faq":
            corpus_dict_faq = {
                key: str(value)
                for key, value in key_to_source_dict.items()
                if key in q_dict["source"]
            }
            retrieved = cohere_retrieve(
                q_dict["query"], q_dict["source"], corpus_dict_faq
            )
            answer_dict["answers"].append({"qid": q_dict["qid"], "retrieve": retrieved})
        else:
            raise ValueError("Something went wrong")  # 如果過程有問題，拋出錯誤

    return answer_dict
