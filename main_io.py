"""IO functions for the main script."""

import os
import json
import pdfplumber  # 用於從PDF文件中提取文字的工具

from tqdm import tqdm


# 載入參考資料，返回一個字典，key為檔案名稱，value為PDF檔內容的文本
def load_data(source_path):
    """Load data from the source path."""
    masked_file_ls = os.listdir(source_path)  # 獲取資料夾中的檔案列表
    corpus_dict = {
        int(file.replace(".pdf", "")): read_pdf(os.path.join(source_path, file))
        for file in tqdm(masked_file_ls)
    }  # 讀取每個PDF文件的文本，並以檔案名作為鍵，文本內容作為值存入字典
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

    return pdf_text  # 返回萃取出的文本


def parse_input(args):
    """Parse the input arguments and return the corresponding data."""
    with open(args.question_path, "rb") as f:
        qs_ref = json.load(f)  # 讀取問題檔案

    source_path_insurance = os.path.join(
        args.source_path, "insurance"
    )  # 設定參考資料路徑
    corpus_dict_insurance = load_data(source_path_insurance)

    source_path_finance = os.path.join(args.source_path, "finance")  # 設定參考資料路徑
    corpus_dict_finance = load_data(source_path_finance)

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
