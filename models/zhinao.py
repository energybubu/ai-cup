from typing import Dict, List, Tuple, Union

import numpy as np
import torch
import transformers
from tqdm import tqdm
from transformers.trainer_pt_utils import LabelSmoother

from models.constant import max_seq_length

IGNORE_TOKEN_ID = LabelSmoother.ignore_index


def preprocess(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    max_len: int = max_seq_length,
    system_message: str = "",
    device=None,
) -> Dict:
    roles = {"user": "<|im_start|>user", "assistant": "<|im_start|>assistant"}
    answer_len = 64

    im_start = tokenizer.im_start_id
    im_end = tokenizer.im_end_id
    nl_tokens = tokenizer("\n").input_ids
    _system = tokenizer("system").input_ids + nl_tokens
    _user = tokenizer("user").input_ids + nl_tokens
    _assistant = tokenizer("assistant").input_ids + nl_tokens

    # Apply prompt templates
    input_ids, targets = [], []
    for i, source in enumerate(sources):
        ## system_message
        input_id, target = [], []
        system = (
            [im_start]
            + _system
            + tokenizer(
                system_message, max_length=max_len - answer_len, truncation=True
            ).input_ids
            + [im_end]
            + nl_tokens
        )
        input_id += system
        target += (
            [im_start] + [IGNORE_TOKEN_ID] * (len(system) - 3) + [im_end] + nl_tokens
        )
        assert len(input_id) == len(target)

        ## query ans
        source = "\n\n".join(source)
        role = "<|im_start|>user"
        _input_id = (
            tokenizer(role, max_length=max_len - answer_len, truncation=True).input_ids
            + nl_tokens
            + tokenizer(
                source, max_length=max_len - answer_len, truncation=True
            ).input_ids
            + [im_end]
            + nl_tokens
        )
        input_id += _input_id
        if role == "<|im_start|>user":
            _target = (
                [im_start]
                + [IGNORE_TOKEN_ID] * (len(_input_id) - 3)
                + [im_end]
                + nl_tokens
            )
        elif role == "<|im_start|>assistant":
            _target = (
                [im_start]
                + [IGNORE_TOKEN_ID]
                * len(
                    tokenizer(
                        role, max_length=max_len - answer_len, truncation=True
                    ).input_ids
                )
                + _input_id[
                    len(
                        tokenizer(
                            role, max_length=max_len - answer_len, truncation=True
                        ).input_ids
                    )
                    + 1 : -2
                ]
                + [im_end]
                + nl_tokens
            )
        else:
            raise NotImplementedError
        target += _target

        ## label use placeholder 0; It will be masked later in the modeling_zhinao.py
        role = "<|im_start|>assistant"
        _input_id = (
            tokenizer(role, max_length=max_len - answer_len, truncation=True).input_ids
            + nl_tokens
            + tokenizer("0", max_length=max_len - answer_len, truncation=True).input_ids
            + [im_end]
            + nl_tokens
        )
        input_id += _input_id
        if role == "<|im_start|>user":
            _target = (
                [im_start]
                + [IGNORE_TOKEN_ID] * (len(_input_id) - 3)
                + [im_end]
                + nl_tokens
            )
        elif role == "<|im_start|>assistant":
            _target = (
                [im_start]
                + [IGNORE_TOKEN_ID]
                * len(
                    tokenizer(
                        role, max_length=max_len - answer_len, truncation=True
                    ).input_ids
                )
                + _input_id[
                    len(
                        tokenizer(
                            role, max_length=max_len - answer_len, truncation=True
                        ).input_ids
                    )
                    + 1 : -2
                ]
                + [im_end]
                + nl_tokens
            )
        else:
            raise NotImplementedError
        target += _target

        assert len(input_id) == len(target)
        input_id += [tokenizer.pad_token_id] * (max_len - len(input_id))
        target += [IGNORE_TOKEN_ID] * (max_len - len(target))
        if len(input_id) > max_len:
            print("max_len_error")
            print(tokenizer.decode(input_id))

        input_ids.append(input_id[:max_len])
        targets.append(target[:max_len])
    input_ids = torch.tensor(input_ids, dtype=torch.int)
    targets = torch.tensor(targets, dtype=torch.int)
    # print(f"input_ids {input_ids.shape}")
    # print(f"targets {targets.shape}")

    return dict(
        input_ids=input_ids.to(device),
        labels=targets.to(device),
        attention_mask=input_ids.ne(tokenizer.pad_token_id).to(device),
    )


class FlagRerankerCustom:
    def __init__(self, model_name_or_path: str = None, use_fp16: bool = False) -> None:
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_name_or_path,
            model_max_length=max_seq_length,
            padding_side="right",
            use_fast=False,
            trust_remote_code=True,
        )
        self.tokenizer.pad_token_id = self.tokenizer.eod_id
        config = transformers.AutoConfig.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            bf16=True,
            max_position_embeddings=max_seq_length,
        )
        config.use_cache = False
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            config=config,
            trust_remote_code=True,
        )
        self.model.linear.bfloat16()

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

    def compute_score(
        self,
        sentence_pairs: Union[List[Tuple[str, str]], Tuple[str, str]],
        batch_size: int = 128,
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
            disable=False,
        ):
            sentences_batch = sentence_pairs[
                start_index : start_index + batch_size
            ]  # [[q,ans],[q, ans]...]
            inputs = preprocess(
                sources=sentences_batch,
                tokenizer=self.tokenizer,
                max_len=max_length,
                device=self.device,
            )
            scores = (
                self.model(**inputs, return_dict=True)
                .logits.view(
                    -1,
                )
                .float()
            )
            all_scores.extend(scores.cpu().numpy().tolist())

        if len(all_scores) == 1:
            return all_scores[0]
        return all_scores


def zhinao_retrieve(qs, source, corpus_dict):
    model_name_or_path = "360Zhinao-1.8B-Reranking"
    model = FlagRerankerCustom(model_name_or_path, use_fp16=False)
    documents = [corpus_dict[id] for id in source]
    pairs = [[qs, doc] for doc in documents]

    scores = model.compute_score(pairs, max_length=2048)
    best_idx = np.argmax(scores)
    return source[best_idx]


def zhinao_rerank(
    qs_ref,
    corpus_dict_insurance,
    corpus_dict_finance,
    key_to_source_dict,
):
    answer_dict = {"answers": []}  # 初始化字典

    for q_dict in qs_ref["questions"]:
        if q_dict["category"] == "finance":
            # 進行檢索
            retrieved = zhinao_retrieve(
                q_dict["query"], q_dict["source"], corpus_dict_finance
            )
            # 將結果加入字典
            answer_dict["answers"].append({"qid": q_dict["qid"], "retrieve": retrieved})

        elif q_dict["category"] == "insurance":
            retrieved = zhinao_retrieve(
                q_dict["query"], q_dict["source"], corpus_dict_insurance
            )
            answer_dict["answers"].append({"qid": q_dict["qid"], "retrieve": retrieved})

        elif q_dict["category"] == "faq":
            corpus_dict_faq = {
                key: str(value)
                for key, value in key_to_source_dict.items()
                if key in q_dict["source"]
            }
            retrieved = zhinao_retrieve(
                q_dict["query"], q_dict["source"], corpus_dict_faq
            )
            answer_dict["answers"].append({"qid": q_dict["qid"], "retrieve": retrieved})

        else:
            raise ValueError("Something went wrong")  # 如果過程有問題，拋出錯誤
    return answer_dict
