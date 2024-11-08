import re

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)


def faq2text(faqs):
    faq_text = []
    for faq in faqs:
        question, answers = faq["question"], faq["answers"]
        text = f"問題：{question}\n"
        if len(answers) == 1:
            text += f"回答：{answers[0]}"
        else:
            text += "回答："
            for i, answer in enumerate(answers):
                text += f"\n{i+1}. {answer}"

        faq_text.append(text)

    return "\n\n".join(faq_text)


def get_model(model_path):
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=BitsAndBytesConfig(load_in_8bit=True),
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )
    return pipe


# pipe = get_model("Qwen/Qwen2.5-7B-Instruct")
def remove_number(text):
    return re.sub(r"\d+", " ", text)


def remove_eng(text):
    return re.sub(r"[a-zA-Z]", " ", text)


def remove_multiple_spaces(text):
    return re.sub(r"\s+", " ", text)


def remove_eng_punctuation(text):
    return re.sub(r"[\(\).,:;\"\'!?#$%&*+-/<=>@^_`{|}~]", " ", text)


def remove_non_used_chars(text):
    return remove_multiple_spaces(
        remove_eng_punctuation(remove_eng(remove_number(text)))
    )


def refine_pdf(text):
    messages = [
        {
            "role": "system",
            "content": "你是一個文檔整理專家，接下來請依照指示，幫我仔細地改寫文檔。",
        },
        {
            "role": "user",
            "content": f"""以下我將提供一篇文檔，請依照下列重點進行改寫，文章改寫後，將會被存進資料庫提供檢索，請務必保留所有重點，並去除所有冗贅內容。
1. 請將文檔中所有沒有意義的數字去除。
2. 請將文檔中所有表格、圖片、圖表等內容去除，並以文字簡短描述其內容。
3. 將網址、聯絡資訊、電子郵件等個人資訊去除。
         
文檔如下：
{text}""",
        },
    ]

    generation_args = {
        "max_new_tokens": 8192,
        "return_full_text": False,
    }
    content = pipe(messages, **generation_args)[0]["generated_text"]
    content = content.split("文檔如下：")[1].strip()
    return content
