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

from data.translate import trad2simp

from models.gpt import gpt_rerank
from models.constant import gpt_rerank_threshold

def qwen_gpt_retrieve(model, qs, source, corpus_dict):
    documents = [corpus_dict[id] for id in source]
    # pairs = [[qs, doc] for doc in documents]
    pairs = [[trad2simp(qs), trad2simp(doc)] for doc in documents]

    scores = model.compute_score(pairs, max_length=512, overlap=400)
    best_idx = np.argmax(scores)

    ## get the second best score
    best_score = scores[best_idx]
    scores[best_idx] = -np.inf
    second_best_idx= np.argmax(scores)
    if best_score - scores[second_best_idx] < gpt_rerank_threshold:
        result = gpt_rerank(qs, documents[best_idx], documents[second_best_idx])
        if result == 'first':
            return source[best_idx]
        elif result == 'second':
            return source[second_best_idx]

    return source[best_idx]


def qwen_gpt_rerank(
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

    model = FlagRerankerCustom(model=model, tokenizer=tokenizer, use_fp16=True)

    for q_dict in tqdm(qs_ref["questions"]):
        if q_dict["category"] == "finance":
            # 進行檢索
            retrieved = qwen_gpt_retrieve(
                model, q_dict["query"], q_dict["source"], corpus_dict_finance
            )
            # 將結果加入字典
            answer_dict["answers"].append({"qid": q_dict["qid"], "retrieve": retrieved})

        elif q_dict["category"] == "insurance":
            retrieved = qwen_gpt_retrieve(
                model, q_dict["query"], q_dict["source"], corpus_dict_insurance
            )
            answer_dict["answers"].append({"qid": q_dict["qid"], "retrieve": retrieved})

        elif q_dict["category"] == "faq":
            corpus_dict_faq = {
                key: str(value)
                for key, value in key_to_source_dict.items()
                if key in q_dict["source"]
            }
            retrieved = qwen_gpt_retrieve(
                model, q_dict["query"], q_dict["source"], corpus_dict_faq
            )
            answer_dict["answers"].append({"qid": q_dict["qid"], "retrieve": retrieved})

        else:
            raise ValueError("Something went wrong")  # 如果過程有問題，拋出錯誤
    return answer_dict
