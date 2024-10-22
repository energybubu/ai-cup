import cohere
from constant import *

def rerank_documents(query, documents, top_n = 3):
    co = cohere.ClientV2(cohere_api_key)
    
    response = co.rerank(
        model = "rerank-multilingual-v3.0",
        query = query,
        documents = documents,
        top_n = top_n,
    )
    
    return response

if __name__ == "__main__":
    query = "比特幣什麼時候會漲？"
    docs = [
        "專家預測加密貨幣會在一個月內上漲。",
        "今天跟朋友去了一家新的餐廳，食物很好吃。",
        "請問您需要什麼服務？",
        "台股今天開盤上漲。",
        "六福村水樂園是一個很好玩的地方。",
    ]
    print(rerank_documents(query, docs))