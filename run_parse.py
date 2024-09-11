from parse import parse_pdf
from processonlyfortable_hit import preprocess
import pymupdf4llm
import re
import os
import warnings
import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI
import fitz
import shapely.geometry as sg
from shapely.geometry.base import BaseGeometry
from shapely.validation import explain_validity
from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env.


def run_parse(pdf_path, is_ocr, ocr_lang,finance,page):
    openai_api_key = os.getenv("OPENAI_API_KEY")
    image_paths,recs, output_dir = parse_pdf(pdf_path,finance=finance,minimun_merge_size=40,merge_distance=60,horizontal_merge_distance=60,near_distance=5,horizontal_near_distance=5,page = page)
    pymupdf_content = pymupdf4llm.to_markdown(pdf_path,page_chunks=True, write_images=False,margins=(0,0,0,0))    
    txt_pages, GPT_CALL_COUNT, image_dict = preprocess(pymupdf_content,image_paths,recs,pdf_path,openai_api_key,output_dir,is_ocr, ocr_lang) #ocr_lang = chinese_cht or en ; md_text,
    return txt_pages, GPT_CALL_COUNT, image_dict

    # print("====================")
    # print("Parse_pdf抓到的內容：", recs[0])
    # print("pymupdf抓到的表格：", pymupdf_content[0]["tables"])
    # print("pymupdf抓到的圖片：",pymupdf_content[0]["images"])
    # print("====================")
    # print("Parse_pdf抓到的內容：", recs[1])
    # print("pymupdf抓到的表格：",pymupdf_content[1]["tables"])
    # print("pymupdf抓到的圖片：",pymupdf_content[1]["images"])




# pdf_document = fitz.open(pdf_path)
# # 保存页面为图片
# p1 =  (35.5625, 524.7321166992188, 239.6875, 640.21875)
# p2 = (35.596153259277344, 28.049999237060547, 311.6538391113281, 802.9500122070312)

# for page_index, page in enumerate(pdf_document):
#     if page_index == 0:
#         rect = fitz.Rect(p1)
#     if page_index == 1:
#         rect = fitz.Rect(p2)
#     pix = page.get_pixmap(clip=rect, matrix=fitz.Matrix(4, 4))
#     name = f'pymupdf表格內容{page_index}.png'
#     pix.save(os.path.join(output_dir, name))

def add_chunk_to_db(chunks, collection):
    # 將 chunks 添加到 Chroma 集合中
    for i, chunk in enumerate(chunks):
        collection.add(
            documents=[chunk],
            metadatas=[{"source": f"chunk_{i}"}],
            ids=[f"id_{i}"]
        )


def query_and_respond(query ,collection, k=3):
    # 從 Chroma 檢索相關文檔
    client = OpenAI()
    results = collection.query(
        query_texts=[query],
        n_results=k
    )

    # 準備 prompt
    context = "\n".join(results['documents'][0])
    prompt = f"""Given the following context and question, please provide a relevant answer. If the context doesn't contain enough information to answer the question, please say so.
                Context: {context}
                Question: {query}
                Answer:"""

    # 使用 GPT 生成回答
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content

def query_and_respond(query ,collection, k=3):

    client = OpenAI()
    results = collection.query(
        query_texts=[query],
        n_results=k
    )

    # 準備 prompt
    context = "\n".join(results['documents'][0])

    prompt = f"""Given the following context and question, please provide a relevant answer. If the context doesn't contain enough information to answer the question, please say so.
                Context: {context}
                Question: {query}
                Answer:"""

    # 使用 GPT 生成回答
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content

def query_only(query ,collection, k=3): 
    client = OpenAI()
    results = collection.query(
        query_texts=[query],
        n_results=k
    )

    # 準備 prompt
    context = "\n".join(results['documents'][0])
    return context

# # 使用範例
# query = "請問A+B+C+D 合計要幾學分?"
# answer = query_and_respond(query)
# print(f"Question: {query}")
# print(f"Answer: {answer}")