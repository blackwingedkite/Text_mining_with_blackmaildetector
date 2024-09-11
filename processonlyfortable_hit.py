import os
import re
import time
from typing import List, Tuple, Optional, Dict
import logging
import ast
from IPython.display import HTML, display
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
from GeneralAgent import Agent
from paddleocr import PaddleOCR, draw_ocr

def preprocess(
        pymupdf4llm_list: List[Dict],
        image_paths: List[str],
        rects:List[List[Tuple]], #座標位置
        pdf_path:str,
        openai_api_key:str,
        output_dir:str,
        is_ocr:bool=False,
        ocr_lang :str='chinese_cht',
        
) -> str:
    def find_pic_images(directory,page_number=0):
        pic_images = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.startswith(str(page_number)+'_') and file.endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif")):
                    pic_images.append(os.path.join(root, file))
        return pic_images


    def filter_table(content:str)-> str:
        table_pattern = re.compile(
        r'(\|(?:[^\n]*\|)+\n'   # 匹配表格头部行
        r'\|(?:\s*[-:]+\s*\|)+\s*\n'  # 匹配表格分隔行
        r'(?:\|(?:[^\n]*\|)\n)+)'  # 匹配表格内容行
        )
        # table_pattern = re.compile(
        # r'(\|(?:[^\n]*?\|)+\n'        # 匹配表格头部行
        # r'(?:\|(?:\s*[-:]+\s*\|)+\s*\n)?'  # 可選的表格分隔行
        # r'(?:\|(?:[^\n]*?\|)\n)+)'    # 匹配表格内容行
        # )
        result = table_pattern.findall(content)
        return result


    def clean_string(text):
        # 移除HTML外部網站連結
        pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        text = re.sub(pattern, '', text)
        # 移除換行符號
        text = text.replace('\n', '')
        return text

    def illustrate_table(table_path, table_markdown_text, ocr_table=False):
        print(f"illustrating table {table_path}, content: {table_markdown_text[0:10]}...")
        agent = Agent(role=table_role_prompt, api_key=openai_api_key, base_url=None, model=model, disable_python_run=False)
        local_prompt = table_local_prompt + table_markdown_text
        content = agent.run([local_prompt, {'image': output_dir+table_path}])
        if not ocr_table:
            final_content_list.append(content)
        else:
            ocr_table_list.append(content)

    def illustrate_image(image_path):
        print(f"illustrating image {image_path}")
        agent = Agent(role=image_prompt, api_key=openai_api_key, base_url=None, model=model, disable_python_run=True)
        content = agent.run([image_prompt, {'image':output_dir+image_path}])
        image_dict[content] = output_dir + image_path
        final_image_list.append(content)

    def check_same_table(table_path, table_markdown_text, ocr_table=False):
        agent = Agent(role=check_same_table_prompt, api_key=openai_api_key, base_url=None, model=model, disable_python_run=False)
        local_prompt = table_checksame_prompt + table_markdown_text
        content = agent.run([local_prompt, {'image': output_dir+table_path}])
        if "true" in content or "True" in content:
            return True
        else:
            return False


    def check_table_or_image(pymupdf_table_list, rect＿list):
        table_to_path = dict()
        # pymu_table[bbox]是座標位置
        for pymu_table in pymupdf_table_list:
            #parse_table也是座標位置
            for table_index, parse_table in enumerate(rect＿list):
                #contract = (pymu_table['bbox'][1] - parse_table[1]) + (pymu_table['bbox'][0]-parse_table[0])
                rect_h_l = (parse_table[2]-parse_table[0])+(parse_table[3]-parse_table[1])
                pymu_h_l = (pymu_table['bbox'][2]-pymu_table['bbox'][0])+(pymu_table['bbox'][3]-pymu_table['bbox'][1])
                if (abs(rect_h_l-pymu_h_l)) < 20 and (abs(parse_table[0]-pymu_table['bbox'][0])) < 10:
                    path = str(page_index) + '_' + str(table_index) +'.png' 
                    table_to_path[str(pymu_table)] = path

        return table_to_path

    def check_row_column(table):
        rows = table.strip().split('\n')
        num_rows = len(rows) - 1  # 減去頭部行和分隔行
        num_cols = len(rows[0].split('|')) - 2  # 減去開頭和結尾的空字符串
        return num_rows, num_cols



    def make_path_to_markdown(table_to_path, table_markdown, GPT_COUNT):
        #現在我們有的東西：
        #1. 表格所有的markdown內容：是filter出來的 List(String)
        #2. 表格的座標位置+行列數：pymu_table[bbox], pymu_table['rows'], pymu_table['cols'] List(Dict)
        #thought:
        #1. OCR解(depr.)
        #2. 對應表格的行列數
        #3. 若行列數相同，看位置(從上到下從左到右)
        #4. GPT護城河(先問是否相同，再進行解析)
        path_to_markdown = dict()
        markdown_to_rc = dict()
        for table in table_markdown:
            num_rows, num_cols = check_row_column(table)
            print(num_rows, num_cols)
            markdown_to_rc[table] = (num_rows, num_cols)

        for content in table_to_path:
            content = ast.literal_eval(content)
            content_rc = (content["rows"], content["columns"])
            content_rc2 = (content["rows"]+1, content["columns"])
            print(content_rc)
            same_list = []
            for markdown in markdown_to_rc:
                if content_rc == markdown_to_rc[markdown] or content_rc2 == markdown_to_rc[markdown] :
                    same_list.append(markdown)
            
            if len(same_list) == 1:
                path_to_markdown[table_to_path[str(content)]] = same_list[0]
                del markdown_to_rc[same_list[0]]
            if len(same_list) > 1:
                for markdown in same_list:
                    GPT_COUNT += 1
                    is_same = check_same_table(table_to_path[str(content)], markdown)
                    if is_same:
                        path_to_markdown[table_to_path[str(content)]] = markdown
                        del markdown_to_rc[markdown]
                        break
        return path_to_markdown
                    


    GPT_COUNT = 0

    model = 'gpt-4o-mini'

    image_prompt = """你是一個具有豐富人類智慧的專業幫手機器人，你會用人類的角度來詮釋你所得到的圖片，請你使用繁體中文對這張圖片進行摘要"""

    table_role_prompt= """
    你現在是一位專注於製作HTML表格的工程師，你的任務是要畫出一個可以顯示的表格，並讓人類容易閱讀。
    """

    table_local_prompt = """
    ### HTML Table Merging Task Instructions:
    You are now an HTML table engineer, and your task is to merge the table structure found in an image with content provided in Markdown format. You must follow these guidelines closely:
    1. **Full HTML Output**: Your final output must be in **complete HTML format**. Do not omit any part of the output, and make sure every necessary HTML tag is included.
    2. **Reference Markdown Content**: Integrate relevant content from the Markdown file as needed, paying attention to how it enhances the table's completeness and clarity.
    3. **Merge Table Cells**: Ensure that any merged cells are fully reflected in the HTML structure, accurately capturing the design of the table.
    4. **Handle Infinite Tables**: If the table structure extends infinitely or contains a significant number of rows/columns, ensure that the structure remains accurate and maintains proper alignment across the entire table.
    5. **Language Consistency**: Answer any prompts or questions **using the language found in the table** (if applicable).
    **Important Reminder**: Ensure that the final table structure matches the format and layout from the image, without any misalignment or formatting errors. 
    PLEASE DO NOT RETURN ANYTHING OTHER THAN THE HTML OUTPUT.
    """
    check_same_table_prompt = """你是一個具有豐富人類智慧的專業幫手機器人，你的工作是辨別這張圖片是否和這個表格相同，請你使用繁體中文回答"""
    table_checksame_prompt = """你將得到一個以markdown格式呈現的表格以及一張圖片，你的任務是要判斷這張圖片是否和這個表格相同
    若你判斷兩者相同，請回傳True
    若你判斷兩者不同，請回傳False
    請不要回傳其他內容，只回傳True或False
    \n\n文字內容:""" 
    # 用來存放每一頁的文字內容
    txt_pages = list()

    # 敘述內容：圖片位置
    image_dict = dict()
        
    # 創建parse_txt資料夾
    if not os.path.isdir("parse_txt"):
        os.makedirs("parse_txt")
    file_name = os.path.basename(pdf_path)
    final_output_path = "parse_txt/"+str(file_name.rstrip(".pdf"))+"_parse.txt"
    with open(final_output_path, "w") as file:
        pass

    if is_ocr:
        ocr = PaddleOCR(use_angle_cls=True, lang=ocr_lang)  #chinese_cht

    #實際上，我們傳進去的圖片內容都是從parse.py裡面得到的圖片，而pymupdf只是輔助
    doc_length = len(pymupdf4llm_list)
    for page_index,page_content in enumerate(pymupdf4llm_list):
        print(f"processing page {page_index+1}, total {doc_length} pages")


        ocr_table_list = []
        final_content_list = []
        final_image_list = []
        

        # 尋找Pymupdf輸出的文字中的markdown部分
        pymupdf_original_text = page_content['text']
        table_markdown = filter_table(pymupdf_original_text)
        print(table_markdown)
        # rect_list: 用parse.py取出的座標位置
        rect＿list = rects[page_index]
        pymupdf_table_list = page_content['tables']
        pymupdf_image_list = page_content['images']
        parse_path_list = [filename for filename in image_paths if filename.startswith(str(page_index)+'_')]

        if len(pymupdf_table_list) > 0:
            print("hello, now we have some tables")
            table_to_path = check_table_or_image(pymupdf_table_list, rect＿list)
            path_to_markdown = make_path_to_markdown(table_to_path, table_markdown, GPT_COUNT)
            # table_to_path : table pymupdf to 圖片path
            # path_to_markdown : 圖片path to文字

            table_path = list(table_to_path.values())
            print("-=-=-=-=-=-=")
            print(table_path)
            print(parse_path_list)
            print(table_to_path)
            print(path_to_markdown)
            print("-=-=-=-=-=-=")
            for image_path in parse_path_list:
                if image_path not in table_path:
                    GPT_COUNT += 1
                    illustrate_image(image_path)

            for path in path_to_markdown:
                GPT_COUNT += 1
                illustrate_table(path, path_to_markdown[path])

        # pymupdf抓到圖片但沒有抓到表格
        elif len(pymupdf_table_list) == 0 and len(pymupdf_image_list)>0:
            for image_path in parse_path_list:
                GPT_COUNT += 1
                illustrate_image(image_path)
            
            
        else: #代表沒有找到表格但是說不定會有擷取到表格圖片，用OCR解
            # 如果pymupdf4llm沒有找到任何東西的話，就把資訊丟到最下面當補充，ＯＣＲ不能夠確定東西在哪裡，檔案另外存
            if not ocr:
                pass
            else:
                res = find_pic_images(output_dir, page_number=page_index)
                if res != []:
                    for image_ in res:
                        ocr_result = ocr.ocr(image_, cls=True)
                        first_chunk = ocr_result[0][0][1][0] 
                        last_chunk = ocr_result[0][-1][1][0]
                        fisrt_find = pymupdf_original_text.find(first_chunk[:20])
                        last_find = pymupdf_original_text.find(last_chunk[len(last_chunk)-20:])
                        # 表格內容
                        if last_find != -1 and fisrt_find != -1: 
                            ocr_text = pymupdf_original_text[fisrt_find:last_find+20]
                            GPT_COUNT += 1
                            illustrate_table(image_, ocr_text, ocr_table=True)
                        else:
                            #當作圖片處理
                            GPT_COUNT += 1
                            illustrate_image(image_)

        output_text = ""

        #若有非OCR的表格，則特別處理
        if len(final_content_list) > 0:
            for num_table, raw_table in enumerate(table_markdown):
                new_page_content = pymupdf_original_text.replace(raw_table, final_content_list[num_table])
                if new_page_content == pymupdf_original_text:
                    # 如果replace没有成功，新增内容到pymupdf_original_text的最下面(目前版本應該都會替代成功)
                    pymupdf_original_text += final_content_list[num_table]
                else:
                    # 如果replace成功，更新pymupdf_original_text
                    pymupdf_original_text = new_page_content
            output_text += clean_string(pymupdf_original_text)

        elif len(final_content_list) == 0:
            output_text += clean_string(pymupdf_original_text)

        # 最後，如果有OCR的表格，放在pymupdf_original_text的最下面
        if len(ocr_table_list) > 0:
            for ocrtable_index, ocr_table in enumerate(ocr_table_list):
                output_text += f"ocr table {ocrtable_index}:"
                output_text += f"{ocr_table}"
                output_text += f"end of ocr table {ocrtable_index}:"
        
        # 最後，如果有圖片，也是放在pymupdf_original_text的最下面
        if len(final_image_list) > 0:
            for index_image, image_content in enumerate(final_image_list):
                output_text += f"image {index_image}:"
                output_text += f"{image_content}"  #將HTML存進去txt
                output_text += f"end of image {index_image}"

        output_dict = {
            "page": page_index+1,
            "text": output_text
        }
        txt_pages.append(output_dict)
        with open(final_output_path, "w") as file:
            file.write(output_text)



    print("\npreprocess done\n")
    return txt_pages, GPT_COUNT, image_dict 