import jieba  # 用於中文文本分詞
from rank_bm25 import BM25Okapi  # 使用BM25演算法進行文件檢索
from tqdm import tqdm
import numpy as np
from typing import Dict, List, Tuple, Union
import numpy as np
import torch
import transformers
from tqdm import tqdm
from transformers import (
    DataCollatorWithPadding,
    PreTrainedModel,
    PreTrainedTokenizer,
    Qwen2Config,
    Qwen2ForSequenceClassification,
)
from transformers.trainer_pt_utils import LabelSmoother
from models.constant import max_seq_length
from models.qwen import FlagRerankerCustom
import time
import cohere
from models.constant import *

# 根據查詢語句和指定的來源，檢索答案
def bm25_retrieve_all(qs, source, corpus_dict):
    """BM25 Retrieve Function."""
    filtered_corpus = [corpus_dict[int(file)] for file in source]

    # [TODO] 可自行替換其他檢索方式，以提升效能

    tokenized_corpus = [
        list(jieba.cut_for_search(doc)) for doc in filtered_corpus
    ]  # 將每篇文檔進行分詞
    bm25 = BM25Okapi(tokenized_corpus)  # 使用BM25演算法建立檢索模型
    tokenized_query = list(jieba.cut_for_search(qs))  # 將查詢語句進行分詞
    ans = bm25.get_top_n(
        tokenized_query, list(filtered_corpus)
    )  # 根據查詢語句檢索，返回最相關的文檔，其中n為可調整項
    assert len(ans) == len(source)
    
    res = [-1] * len(source)
    for i, a in enumerate(ans):
        for key, value in enumerate(source):
            if corpus_dict[value] == a:
                res[key] = i + 1
                break

    assert -1 not in res
    return res

def qwen_retrieve_all(model, qs, source, corpus_dict):
    documents = [corpus_dict[id] for id in source]
    pairs = [[qs, doc] for doc in documents]

    scores = model.compute_score(pairs, max_length=2048)
    reanks = zip(range(1, len(scores) + 1), scores)
    reanks = sorted(reanks, key=lambda x: x[1], reverse=True)
    return [r[0] for r in reanks]

def cohere_retrieve_all(qs, source, corpus_dict):
    co = cohere.ClientV2(cohere_api_key)

    documents = []
    for file_num in source:
        documents.append(corpus_dict[file_num])

    response = co.rerank(
        model="rerank-multilingual-v3.0", query=qs, documents=documents
    )

    time.sleep(cohere_api_wait_time)

    result = [-1] * len(source)
    for i, res in enumerate(response.results):
        result[res.index] = i + 1

    assert -1 not in result

    return result

def hybrid_retrieve(qwen_model, qs, source, corpus_dict, k=60):
    # bm25_result = bm25_retrieve_all(qs, source, corpus_dict)
    qwen_result = qwen_retrieve_all(qwen_model, qs, source, corpus_dict)
    cohere_result = cohere_retrieve_all(qs, source, corpus_dict)
    rrf_scores = [0] * len(source)
    
    for result in [cohere_result, qwen_result]:
        for i, rank in enumerate(result):
            rrf_scores[i] += 1 / (k + rank)

    best_ind = np.argmax(rrf_scores)
    return source[best_ind]

def hybrid_rerank(
    qs_ref,
    corpus_dict_insurance,
    corpus_dict_finance,
    key_to_source_dict,
):
    answer_dict = {"answers": []}  # 初始化字典
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        "neofung/LdIR-Qwen2-reranker-1.5B",
        padding_side="right",
        model_max_length=max_seq_length,
    )

    config = Qwen2Config.from_pretrained(
        "neofung/LdIR-Qwen2-reranker-1.5B",
        trust_remote_code=True,
        bf16=True,
        max_position_embeddings=max_seq_length,  # Ensure the model's config allows 2048 tokens
    )

    model = Qwen2ForSequenceClassification.from_pretrained(
        "neofung/LdIR-Qwen2-reranker-1.5B",
        config=config,
        trust_remote_code=True,
    )

    qwen_model = FlagRerankerCustom(model=model, tokenizer=tokenizer, use_fp16=True)

    for q_dict in tqdm(qs_ref["questions"]):
        if q_dict["category"] == "finance":
            # 進行檢索
            retrieved = hybrid_retrieve(
                qwen_model, q_dict["query"], q_dict["source"], corpus_dict_finance
            )
            # 將結果加入字典
            answer_dict["answers"].append({"qid": q_dict["qid"], "retrieve": retrieved})

        elif q_dict["category"] == "insurance":
            retrieved = hybrid_retrieve(
                qwen_model, q_dict["query"], q_dict["source"], corpus_dict_insurance
            )
            answer_dict["answers"].append({"qid": q_dict["qid"], "retrieve": retrieved})

        elif q_dict["category"] == "faq":
            corpus_dict_faq = {
                key: str(value)
                for key, value in key_to_source_dict.items()
                if key in q_dict["source"]
            }
            retrieved = hybrid_retrieve(
                qwen_model, q_dict["query"], q_dict["source"], corpus_dict_faq
            )
            answer_dict["answers"].append({"qid": q_dict["qid"], "retrieve": retrieved})

        else:
            raise ValueError("Something went wrong")  # 如果過程有問題，拋出錯誤
    return answer_dict


if __name__ == '__main__':
    print(cohere_retrieve_all('我要買保險', [1, 2, 3], {1: '我要買羽毛球', 2: '我要買金融保險', 3: '哈嘍'}))