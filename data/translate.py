import opencc

def trad2simp(text):
    converter = opencc.OpenCC('t2s') 

    converted_text = converter.convert(text)

    return converted_text