"""Implement GPT Rerank Function."""

from dotenv import load_dotenv
from openai import OpenAI

SYSTEM_MSG = """
你是一位華南銀行的客服。
"""

COMPARE_TEMPLATE = """
給定一個客戶的詢問，請決定哪一篇文章更適合回答客戶的問題。

詢問：「{question}」

文章A：「{doc1}」

文章B：「{doc2}」

若文章A更適合回答客戶的問題，請輸出一個字母「A」；若文章B更適合回答客戶的問題，請輸出一個字母「B」。不要輸出多餘的資訊。
""".strip()


def gpt_rerank(question, doc1, doc2):
    """Implement GPT Rerank Function."""
    load_dotenv()

    client = OpenAI()

    user_message = COMPARE_TEMPLATE.format(question=question, doc1=doc1, doc2=doc2)
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_MSG},
            {"role": "user", "content": user_message},
        ],
    )

    response = completion.choices[0].message.content
    assert ("A" in response) ^ ("B" in response)

    result1 = "A" in response

    user_message = COMPARE_TEMPLATE.format(question=question, doc1=doc2, doc2=doc1)
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_MSG},
            {"role": "user", "content": user_message},
        ],
    )

    response = completion.choices[0].message.content
    assert ("A" in response) ^ ("B" in response)

    result2 = "B" in response

    if result1 and result2:
        return "first"
    elif not result1 and not result2:
        return "second"
    else:
        return "equal"
