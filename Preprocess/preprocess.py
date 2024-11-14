"""Preprocess data."""


def faq2text(faqs):
    """Convert FAQ to text."""
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
