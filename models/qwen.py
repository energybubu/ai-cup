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

from data.translate import *

IGNORE_TOKEN_ID = LabelSmoother.ignore_index


def preprocess(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    max_len: int = 1024,
) -> Dict:
    # Apply prompt templates
    input_ids, attention_masks = [], []
    for i, source in enumerate(sources):
        messages = [{"role": "user", "content": "\n\n".join(source)}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        model_inputs = tokenizer([text])
        input_id = model_inputs["input_ids"][0]
        attention_mask = model_inputs["attention_mask"][0]
        if len(input_id) > max_len:
            ## last five tokens: <|im_end|>(151645), \n(198), <|im_start|>(151644), assistant(77091), \n(198)
            diff = len(input_id) - max_len
            input_id = input_id[: -5 - diff] + input_id[-5:]
            attention_mask = attention_mask[: -5 - diff] + attention_mask[-5:]
            assert len(input_id) == max_len
        input_ids.append(input_id)
        attention_masks.append(attention_mask)

    return dict(input_ids=input_ids, attention_mask=attention_masks)


class FlagRerankerCustom:
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        use_fp16: bool = False,
    ) -> None:
        self.tokenizer = tokenizer
        self.model = model
        self.data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
            use_fp16 = False
        if use_fp16:
            self.model.half()

        self.model = self.model.to(self.device)

        self.model.eval()

        self.num_gpus = torch.cuda.device_count()
        if self.num_gpus > 1:
            print(f"----------using {self.num_gpus}*GPUs----------")
            self.model = torch.nn.DataParallel(self.model)

    @torch.no_grad()
    def compute_score(
        self,
        sentence_pairs: Union[List[Tuple[str, str]], Tuple[str, str]],
        batch_size: int = 64,
        max_length: int = 1024,
    ) -> List[float]:
        if self.num_gpus > 0:
            batch_size = batch_size * self.num_gpus

        assert isinstance(sentence_pairs, list)
        if isinstance(sentence_pairs[0], str):
            sentence_pairs = [sentence_pairs]

        all_scores = []
        for start_index in tqdm(
            range(0, len(sentence_pairs), batch_size),
            desc="Compute Scores",
            disable=True,
        ):
            sentences_batch = sentence_pairs[start_index : start_index + batch_size]
            inputs = preprocess(
                sources=sentences_batch, tokenizer=self.tokenizer, max_len=max_length
            )
            inputs = [dict(zip(inputs, t)) for t in zip(*inputs.values())]
            inputs = self.data_collator(inputs).to(self.device)
            scores = self.model(**inputs, return_dict=True).logits
            scores = scores.squeeze()
            all_scores.extend(scores.detach().to(torch.float).cpu().numpy().tolist())

        if len(all_scores) == 1:
            return all_scores[0]
        return all_scores


def qwen_retrieve(model, qs, source, corpus_dict):
    documents = [corpus_dict[id] for id in source]
    # pairs = [[qs, doc] for doc in documents]
    pairs = [[trad2simp(qs), trad2simp(doc)] for doc in documents]

    scores = model.compute_score(pairs, max_length=2048)
    best_idx = np.argmax(scores)
    return source[best_idx]


def qwen_rerank(
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
            retrieved = qwen_retrieve(
                model, q_dict["query"], q_dict["source"], corpus_dict_finance
            )
            # 將結果加入字典
            answer_dict["answers"].append({"qid": q_dict["qid"], "retrieve": retrieved})

        elif q_dict["category"] == "insurance":
            retrieved = qwen_retrieve(
                model, q_dict["query"], q_dict["source"], corpus_dict_insurance
            )
            answer_dict["answers"].append({"qid": q_dict["qid"], "retrieve": retrieved})

        elif q_dict["category"] == "faq":
            corpus_dict_faq = {
                key: str(value)
                for key, value in key_to_source_dict.items()
                if key in q_dict["source"]
            }
            retrieved = qwen_retrieve(
                model, q_dict["query"], q_dict["source"], corpus_dict_faq
            )
            answer_dict["answers"].append({"qid": q_dict["qid"], "retrieve": retrieved})

        else:
            raise ValueError("Something went wrong")  # 如果過程有問題，拋出錯誤
    return answer_dict
