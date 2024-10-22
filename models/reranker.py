import cohere
from models.constant import *
import time

def reranker(qs, source, corpus_dict):
    co = cohere.ClientV2(cohere_api_key)
    
    documents = []
    for file_num in source:
        documents.append(corpus_dict[file_num])

    response = co.rerank(
        model = "rerank-multilingual-v3.0",
        query = qs,
        documents = documents,
        top_n = 1
    )

    time.sleep(reranker_wait_time)
    
    return source[response.results[0].index]