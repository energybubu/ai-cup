"""Zpoint Model."""

import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def zpoint_retrieve(query, source, corpus_dict):
    """Zpoint Retrieve Function."""
    filtered_corpus = [corpus_dict[int(file)] for file in source]
    model = SentenceTransformer("iampanda/zpoint_large_embedding_zh")
    # 將文檔轉換為向量
    corpus_embeddings = model.encode(
        filtered_corpus, normalize_embeddings=True, convert_to_tensor=True
    )
    # 將查詢語句轉換為向量
    query_embedding = model.encode(
        query, normalize_embeddings=True, convert_to_tensor=True
    )
    # 計算查詢語句與文檔的相似度
    cos_scores = torch.nn.functional.cosine_similarity(
        query_embedding, corpus_embeddings
    )
    # 找出最相似的文檔
    top_results = torch.topk(cos_scores, k=1)
    return source[top_results.indices[0]]


def zpoint_rerank(
    qs_ref,
    corpus_dict_insurance,
    corpus_dict_finance,
    key_to_source_dict,
):
    """Zpoint Rerank Function."""
    answer_dict = {"answers": []}  # 初始化字典

    for q_dict in tqdm(qs_ref["questions"]):
        if q_dict["category"] == "finance":
            # 進行檢索
            retrieved = zpoint_retrieve(
                q_dict["query"], q_dict["source"], corpus_dict_finance
            )
            # 將結果加入字典
            answer_dict["answers"].append({"qid": q_dict["qid"], "retrieve": retrieved})

        elif q_dict["category"] == "insurance":
            retrieved = zpoint_retrieve(
                q_dict["query"], q_dict["source"], corpus_dict_insurance
            )
            answer_dict["answers"].append({"qid": q_dict["qid"], "retrieve": retrieved})

        elif q_dict["category"] == "faq":
            corpus_dict_faq = {
                key: str(value)
                for key, value in key_to_source_dict.items()
                if key in q_dict["source"]
            }
            retrieved = zpoint_retrieve(
                q_dict["query"], q_dict["source"], corpus_dict_faq
            )
            answer_dict["answers"].append({"qid": q_dict["qid"], "retrieve": retrieved})

        else:
            raise ValueError("Something went wrong")  # 如果過程有問題，拋出錯誤
    return answer_dict
