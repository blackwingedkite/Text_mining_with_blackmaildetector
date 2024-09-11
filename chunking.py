import re
import os
from GeneralAgent import Agent
from dotenv import load_dotenv
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
model = "gpt-4o-mini"

role_prompt = "你現在是一個專業的機器人，可以針對HTML內容進行分析，並且提供相關的建議。"
whether_to_merge_prompt = """
Carefully examine the two provided tables and determine whether they should be merged into a single table.
If you believe they should be merged, answer "True". If not, answer "False". Do not output any other text.

Consider the following:
1. If the tables have partially similar content but some differences, yet are about the same topic, output "True".
2. If the tables have completely different content or are about different topics, output "False".
3. Observe the table titles, content, and format to make your decision.
4. If the first table has a title but the second doesn't, there's a higher chance they're part of the same table.
5. Look for <thead> and </thead> tags to aid your judgment.
6. Consider the feasibility and difficulty of merging when making your decision. This point is quite important because you will be asked to merge the tables if you answer "True".

Your decision will determine whether we proceed with merging (if "True") or keep the tables separate (if "False").
This decision is crucial and it very important to my career. 

Please choose carefully between "True" and "False" only, without any additional text.

"""
merging_prompt = """
**Instructions for Merging Two HTML Tables into One Coherent, Valid HTML Table**

### Important Notes:

1. **Table Title**: If one table has a title and the other doesn’t, use the existing title as the title for the final merged table. **The table's title is critical, ensure it's clear, accurate, and prominent. Readibility is important too**
2. **Unique Data**: Combine the content from both tables, preserving all unique information from each.
3. **Unused or Redundant Content**: Remove any unused or redundant content to ensure the final table is concise and easy to read.
4. **Reconcile Structures**: If the tables have different column structures, reconcile them logically. Ensure that each column serves a clear purpose and that all data fits appropriately.
5. **HTML Validity**: Ensure the final table is well-formatted, semantically correct, and maintains full HTML validity.
6. **Preserve Styling**: Retain any existing CSS classes or styles attached to the table, rows, or cells.
7. **Semantic Structure**: Ensure that the `<thead>`, `<tbody>`, and `<tfoot>` tags are used properly, reflecting the structure of the merged content.
8. **No Duplicate Data**: Avoid displaying duplicate information. If similar data appears in both tables, ensure only the most recent or comprehensive information is included.
9. **All Data Preserved**: Ensure that all content in the original tables, especially important details, is preserved. **You must not remove any vital information, even if the data looks similar.**

### Final Deliverable:

- Provide the HTML code for the **merged table**, formatted and indented properly for readability. Notice the rationality of title ans structure.
- Ensure the table is ready for immediate use in an HTML document.
- don't response anything not html code, like "### Key Points:", "Explanation", and so on .
"""
def gpt_should_merge(upper_table, lower_table):
    agent = Agent(role=role_prompt, api_key=openai_api_key, base_url=None, model=model, disable_python_run=False)
    local_prompt = whether_to_merge_prompt + "Table 1:\n" + upper_table+ "Table 1:\n" + lower_table
    content = agent.run([local_prompt]) # , {'image': output_dir+table_path}可以考慮放原始照片
    print("GPT decision is: ", content)
    if "True" in content or "true" in content:
        return True
    else:
        return False

def gpt_start_merge(upper_table, lower_table):
    agent = Agent(role=role_prompt, api_key=openai_api_key, base_url=None, model=model, disable_python_run=False)
    local_prompt = merging_prompt + "Table 1:\n" + upper_table+ "Table 2:\n" + lower_table
    content = agent.run([local_prompt]) # , {'image': output_dir+table_path}可以考慮放原始照片
    return content
    #return upper_table+"\n\n以上是uppertable, 以下是lowertable\n\n"+lower_table

def normalize_page(page_info):
    if page_info["start_page"] == "Not Found":
        page_info["start_page"] = 999999999
        page_info["end_page"] = -1
    return page_info


def merge_chunk_content_new(txt_pages_dict, chunk_size=500,merge_tables = True):
    chunks = []
    html_content = []
    text = "".join([page["text"] for page in txt_pages_dict])
    
    table_pattern = [
    (r'(?:<html.*?>.*?</html>|<table.*?>.*?</table>)', 'table'),
    (r'image \d+:.*?end of image \d', 'image'),
    ]
    for pattern, content_type in table_pattern:
        for match in re.finditer(pattern, text, re.DOTALL | re.IGNORECASE):
            html_content.append((match.start(), match.end(), content_type, match.group()))
    html_content.sort(key=lambda x: x[0])
    # 用於存儲唯一的表格信息
    matches = []
    
    for i, content in enumerate(html_content):
        is_unique = True
        for j, other_content in enumerate(html_content):
            if i != j and content[0] >= other_content[0] and content[1] < other_content[1]:
                # 這個表格是另一個表格的子集，所以它不是唯一的
                is_unique = False
                break
        if is_unique:
            matches.append(content)
    
    for i, content in enumerate(matches, 1):
        print(f"content {i}:")
        print(content)
        print("-" * 50)

    # 從這裡開始合併
    if merge_tables:
        merge_flag_list = []
        # Step 1:檢查這筆資料和下筆資料的距離
        for index, current_item in enumerate(matches):
            # 檢查與下一個區間的距離
            if index >= len(matches) - 1:
                is_close = False  # 最後一筆數據永遠是 False
            elif current_item[2] == "image" or matches[index+1][2] == "image":
                is_close = False
            else:
                next_item = matches[index+1]
                distance_to_next = next_item[0] - current_item[1]
                is_close = distance_to_next < 50
            original_tuple = matches[index]
            add_bool = (is_close,)
            new_tuple = original_tuple + add_bool
            merge_flag_list.append(new_tuple)

        # Step 2: 維持一張diction, 放改前(key)和改後(value)
        # 如果不需改動：value設定為相同
        # 如果需要改動：前者（合併的）value改成gpt處理結果，後者（被合併的）value改成空
        def gpt_merge_table(upper_table, lower_table, continuity, combine_amount):
            #combine amount預設是1，若最後沒有合併則為0
            # 2-1. binary: 是否合併, true: 繼續合併, false: 不合併
            can_merge = gpt_should_merge(upper_table, lower_table)
            if not can_merge:
                combine_amount -= 1
                matches_item[merge_flag_list[index][3]] = merge_flag_list[index][3] #相同
            else:
                # 2-2. 如果判定需要合併，則合併
                final_result = gpt_start_merge(upper_table, lower_table)
                
                matches_item[merge_flag_list[index][3]] = final_result
                matches_item[merge_flag_list[index+combine_amount][3]] = "" # "我已經被合併了"
                
                if continuity:
                    combine_amount += 1
                    gpt_merge_table(final_result, merge_flag_list[index+combine_amount][3],merge_flag_list[index+combine_amount][-1], combine_amount)
            return combine_amount
            

        matches_item = dict()
        combine_amount = 0
        for index, current_item in enumerate(merge_flag_list):
            if combine_amount >0:       
                combine_amount -= 1
                continue
            if current_item[-1] == True:
                upper_tuple = merge_flag_list[index]
                lower_tuple = merge_flag_list[index+1]
                combine_amount = 1
                combine_amount = gpt_merge_table(upper_tuple[3], lower_tuple[3],lower_tuple[-1], combine_amount)
            else:
                matches_item[merge_flag_list[index][3]] = merge_flag_list[index][3]

        # step 3: 存檔新的這個表格涵蓋的頁數範圍
        # 原本的表格只會在一頁中，但是新的表格可能會跨頁
        # 建立新的表格，說這個表格會跨幾頁

        # step 3-1 先建立original_table 對page的字典
        table_page_dic = dict()
        for index, i in enumerate(matches_item):
            #i是原本的文本，理論上可以對應
            table_page_dic[i] = []
            for page in txt_pages_dict:
                if i in page["text"]:
                    table_page_dic[i].append(page["page"])
                    break
            # 如果沒有找到的保險性補齊措施
            if table_page_dic[i] == []:
                if index == 0:
                    table_page_dic[i].append(0)
                else:
                    table_page_dic[i].append(table_page_dic[i-1][-1])
            index += 1

        # step 3-2接者建立新的表格對page的字典
        new_table_page_dic = dict()
        last_content = ""
        

        for index, i in enumerate(matches_item):
            if matches_item[i] == i:
                new_table_page_dic[matches_item[i]] = table_page_dic[i]
            elif matches_item[i] == "": #我已經被合併了
                new_table_page_dic[last_content].extend(table_page_dic[i]) 
            else:
                last_content = matches_item[i]
                new_table_page_dic[last_content] = table_page_dic[i]

    # 接者使用原本的內容處理，最後再額外處理合併表格的部分
    start_checkpoint = []
    end_checkpoint = []
    for i in matches:
        start_checkpoint.append(i[0])
        end_checkpoint.append(i[1])
        #(2725, 2924, 'image', 'image 0.........)

    chunk_start = 0
    chunk_end = 0
    match_item = 0
    while True:
        if match_item < len(matches):
            if chunk_start + chunk_size >= len(text):
                print(str(chunk_start)+ " "+str(len(text)) )
                chunks.append(text[chunk_start:])
                break
            elif chunk_start + chunk_size <= start_checkpoint[match_item]:
                chunk_end = chunk_start + chunk_size
                print(str(chunk_start)+ " "+str(chunk_end))
                chunks.append(text[chunk_start:chunk_end])
                chunk_start = chunk_end
            elif chunk_start + chunk_size > start_checkpoint[match_item]:
                chunks.append(text[chunk_start:start_checkpoint[match_item]])
                print(str(chunk_start)+ " "+str(start_checkpoint[match_item]))
                chunks.append(text[start_checkpoint[match_item]:end_checkpoint[match_item]])
                print(str(start_checkpoint[match_item])+ " "+str(end_checkpoint[match_item]))
                chunk_start = end_checkpoint[match_item]
                chunk_end = end_checkpoint[match_item]
                match_item += 1
        else:
            if chunk_start + chunk_size >= len(text):
                print(str(chunk_start)+ " "+str(len(text)) )
                chunks.append(text[chunk_start:])
                break
            else:
                chunk_end = chunk_start + chunk_size
                print(str(chunk_start)+ " "+str(chunk_end))
                chunks.append(text[chunk_start:chunk_end])
                chunk_start = chunk_end


    #標注頁碼
    now_page = 0
    chunk_page = list()
    print("Number of chunks:", len(chunks))
    print("Number of pages in txt_pages_dict:", len(txt_pages_dict))
    for chunk in chunks:
        inner_dict = {"chunk": chunk, "start_page": "", "end_page":""}

        #first_content = 有可能的第一頁的內容
        #last_content = 有可能的最後一頁的內容
        #content = 這之間的內容（可能跨多頁）
        first_content = txt_pages_dict[now_page]["text"]
        content = first_content
        last_content = ""
        forward_step = 0
        page_loc = []
        while now_page < len(txt_pages_dict): #從第一頁到最後一頁
            if (chunk in last_content) and (chunk not in first_content):
                #所有的內容不在原本這頁了，但是在下一頁．這之後就完全忽略前一頁的內容
                now_page += 1
                page_loc.append(now_page +1)
                break
            elif (chunk in content):
                #跨多頁的內容（含第一頁到最後一頁）會在這裡處理
                end_page = now_page + forward_step
                for step in range(now_page, end_page+1):
                    page_loc.append(step+1)
                if last_content != "":
                    now_page = now_page + forward_step # - 1
                break # 找到 chunk 所在的頁面後退出循環
            else:
                #如果在目前的content找不到，就往下一頁找
                forward_step += 1
                #如果已經到最後一頁了，就不再往下找
                if now_page + forward_step >= len(txt_pages_dict):
                    break
                last_content = txt_pages_dict[now_page+forward_step]["text"]
                content += last_content
        
        if len(page_loc) == 0:
            print("Not Found")
            page_txt = "Not Found"
            inner_dict["start_page"] = page_txt
            inner_dict["end_page"] = page_txt
        else:
            inner_dict["start_page"] = page_loc[0]
            inner_dict["end_page"] = page_loc[-1] 
        chunk_page.append(inner_dict)



    # 我想要改成先進行chunk的編號再進行合併

    merged_chunks_page = []
    for current_chunk, next_chunk in zip(chunk_page[:-1], chunk_page[1:]):
        # Normalize page info
        current_chunk = normalize_page(current_chunk)
        next_chunk = normalize_page(next_chunk)
        
        # Merge content and calculate page range
        merge_content = current_chunk["chunk"] + next_chunk["chunk"]
        start_page = min(current_chunk["start_page"], next_chunk["start_page"])
        end_page = max(current_chunk["end_page"], next_chunk["end_page"])
        
        # Create merged dictionary and append
        merged_chunks_page.append({
            "chunk": merge_content,
            "start_page": start_page,
            "end_page": end_page
        })

        # if ("end of image" in chunks[i]) and ("end of image" not in chunks[i+1]):
        #     merged_chunks.append(chunks[i])
        #     merged_chunks.append(chunks[i+1])
        # elif ("end of image" in chunks[i]) and ("end of image" in chunks[i+1]):
        #     merged_chunks.append(chunks[i])
        # else:
        #     merged_chunks.append(chunks[i] + chunks[i+1])
        # if i==(len(chunks)-2) and ("end of image" in chunks[i+1]):
        #     merged_chunks.append(chunks[i+1])

    # merge_flag_list: (start, end, type, content, is_close)
    # matches_item (dict): 舊對新
    # table_page_dic (dict): 舊對page(list)
    # new_table_page_dic(dict): 新對page(list)
    if merge_tables:
        #小心每個表格都會出現至少兩次
        for dic in merged_chunks_page:
            for count in range(len(new_table_page_dic)):
                if merge_flag_list[count][3] in dic["chunk"]:
                   
                    old_page = merge_flag_list[count][3]
                    new_page = matches_item[merge_flag_list[count][3]] #3ˋ種型態：不變、變長、歸零
                    dic["chunk"] = dic["chunk"].replace(old_page, new_page)


                    possible_start_page = table_page_dic[old_page]
                    possible_start_page.append(dic["start_page"])
                    dic["start_page"] = min(possible_start_page)

                    possible_end_page = list()
                    if new_page != "":
                        possible_end_page.extend(new_table_page_dic[new_page])
                    possible_end_page.append(dic["end_page"])
                    dic["end_page"] = max(possible_end_page)                    

    for dic in merged_chunks_page:
        if dic["chunk"] == "":
            merged_chunks_page.remove(dic)
            continue
        print(dic["chunk"])
        print(str(dic["start_page"]) + " " + str(dic["end_page"]))


    return merged_chunks_page
