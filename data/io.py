"""IO functions for the main script."""

import json
import os

import pdfplumber  # 用於從PDF文件中提取文字的工具
from tqdm import tqdm

CACHE_DIR = ".cache/"

def read_cache(doc_id, category):
    cache_path = os.path.join(CACHE_DIR, f"{category}/{doc_id}.txt")
    if os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            return f.read()

    return None

def save_cache(pdf_text, doc_id, category):
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(os.path.join(CACHE_DIR, category), exist_ok=True)
    
    cache_path = os.path.join(CACHE_DIR, f"{category}/{doc_id}.txt")
    with open(cache_path, "w") as f:
        f.write(pdf_text)

# 載入參考資料，返回一個字典，key為檔案名稱，value為PDF檔內容的文本
def load_data(source_path, use_cache):
    """Load data from the source path."""
    masked_file_ls = os.listdir(source_path)  # 獲取資料夾中的檔案列表

    corpus_dict = {}
    for file in tqdm(masked_file_ls):
        doc_id = int(file.replace(".pdf", ""))
        category = "insurance" if "insurance" in source_path else "finance"
        if use_cache and (cached_pdf_text := read_cache(doc_id, category)):
            pdf_text = cached_pdf_text
        else:
            pdf_text = read_pdf(os.path.join(source_path, file))

        corpus_dict[doc_id] = pdf_text

        save_cache(pdf_text, doc_id, category)

    return corpus_dict


# 讀取單個PDF文件並返回其文本內容
def read_pdf(pdf_loc, page_infos: list = None):
    """Read the PDF file and return its text content."""
    pdf = pdfplumber.open(pdf_loc)  # 打開指定的PDF文件

    # TODO: 可自行用其他方法讀入資料，或是對pdf中多模態資料（表格,圖片等）進行處理

    # 如果指定了頁面範圍，則只提取該範圍的頁面，否則提取所有頁面
    pages = pdf.pages[page_infos[0] : page_infos[1]] if page_infos else pdf.pages
    pdf_text = ""
    for _, page in enumerate(pages):  # 迴圈遍歷每一頁
        text = page.extract_text()  # 提取頁面的文本內容
        if text:
            pdf_text += text
    pdf.close()  # 關閉PDF文件

    pdf_text = pdf_text.replace('\n', '')

    return pdf_text  # 返回萃取出的文本


def parse_input(args):
    """Parse the input arguments and return the corresponding data."""
    with open(args.question_path, "rb") as f:
        qs_ref = json.load(f)  # 讀取問題檔案

    source_path_insurance = os.path.join(
        args.source_path, "insurance"
    )  # 設定參考資料路徑
    corpus_dict_insurance = load_data(source_path_insurance, args.use_cache)

    source_path_finance = os.path.join(args.source_path, "finance")  # 設定參考資料路徑
    corpus_dict_finance = load_data(source_path_finance, args.use_cache)

    with open(os.path.join(args.source_path, "faq/pid_map_content.json"), "rb") as f_s:
        key_to_source_dict = json.load(f_s)  # 讀取參考資料文件
        key_to_source_dict = {
            int(key): value for key, value in key_to_source_dict.items()
        }

    return qs_ref, corpus_dict_insurance, corpus_dict_finance, key_to_source_dict


def write_answer_to_json(args, answer_dict):
    """Write the answer dictionary to a JSON file."""
    # 將答案字典保存為json文件
    with open(args.output_path, "w", encoding="utf8") as f:
        json.dump(
            answer_dict, f, ensure_ascii=False, indent=4
        )  # 儲存檔案，確保格式和非ASCII字符
