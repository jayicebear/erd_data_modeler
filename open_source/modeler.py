
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
import os
import re 
import sys
import ast
from huggingface_hub import login

sys.path.append('/home/ljm/web_modeler')
from Rag.retrieve import query_chroma
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  
# HuggingFace 로그인
login(token="")

def model_load(model_name):
    if model_name == 'google/gemma-3-4b-it':
        tokenizer = AutoProcessor.from_pretrained(model_name) 
    else:  
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    return model, tokenizer


def build_prompt(tables,columns,business_desc, doc):
    prompt = f"""
당신은 데이터 거버넌스 전문가입니다.
업무 설명을 분석하여 관련 테이블과 컬럼을 추천하고, 데이터 모델링을 생성하세요.
회사 내부 자료는 모델링시 참고하세요.
## 사용 가능한 테이블 메타데이터:
{tables}

## 사용 가능한 컬럼 메타데이터:
{columns}

## 업무 설명:
{business_desc}

## 회사 내부 자료:
{doc}

## 요구사항:
1. 사용가능한 테이블, 사용가능한 컬럼, 업무설명을 보고 
SQL 로 데이터 모델링을 하세요 
2. 샘플 데이터 추가 하지 마세요.
** SQL 형식으로만 출력하세요. 설명은 포함하지 마세요.**
"""
    return prompt

def need_question(model_choice,prompt_mode, business_desc,tables,columns): 
    prompt = f"""
    당신은 데이터 모델링 및 데이터 거버넌스 전문가입니다.  
    아래의 **테이블 메타데이터**, **컬럼 메타데이터**, **업무 설명**을 분석하여  
    현재 모델링에 필요한 정보 중 **부족하거나 추가로 알아야 할 개념/데이터/업무요소**를 찾아주세요.  

    이 단계는 RAG 검색에 사용할 키워드를 생성하기 위한 것입니다.  
    
    ---
    ### 테이블 메타데이터
    {tables}

    ### 컬럼 메타데이터
    {columns}

    ### 업무 설명
    {business_desc}

    ### 출력 형식
    - 데이터 모델링에 필요한 추가 정보나 개념을 **짧은 단어 또는 구 형태**로만 출력하세요.
    - 출력은 반드시 **리스트(list)** 형태로 작성하세요.
    - **불필요한 문장, 설명, 이유**는 포함하지 마세요.
    - 출력 5개 이하로 하세요.

    예시 출력:
    ["거래종류코드", "신용정보등급", "고객상세테이블", "거래분류프로세스", "데이터품질진단"]
    """
    if model_choice.lower() == 'qwen':
        result = generate_data_model_qwen(model,tokenizer,prompt,prompt_mode)
    elif model_choice.lower() == 'hyper':
        result = generate_data_model_hyper(model,tokenizer,prompt,prompt_mode)
    elif model_choice.lower() == 'llama':
        result = generate_data_model_llama(model,tokenizer,prompt,prompt_mode)
    elif model_choice.lower() == 'gemma':
        result = generate_data_model_gemma(model,tokenizer,prompt,prompt_mode)
    return result
    
    
def generate_data_model_qwen(model,tokenizer,prompt,prompt_mode):
    # load the tokenizer and the model
    messages = [
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    #print('-------------------------')
    #print(text)
    #print('-------------------------')
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # conduct text completion
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=15000
    )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

    content = tokenizer.decode(output_ids, skip_special_tokens=True)
    content = content.replace('content:','')

    #print("content:", content)
    return content 

def generate_data_model_hyper(model,tokenizer,prompt,prompt_mode):
    chat = [
    {"role": "tool_list", "content": ""},
    {"role": "system", "content": "You are an expert data model analyst."},
    {"role": "user", "content": prompt},
]

    inputs = tokenizer.apply_chat_template(chat, add_generation_prompt=True, return_dict=True, return_tensors="pt")
    inputs = inputs.to("cuda")
    output_ids = model.generate(
        **inputs,
        max_length=10000,
        stop_strings=["<|endofturn|>", "<|stop|>"],
        tokenizer=tokenizer
        )
    result = tokenizer.batch_decode(output_ids)[0]
    
    if prompt_mode.lower() =='rag':    
        result = result.split('<|im_start|>assistant')[1].split('<|im_end|>')[0]
        result = f"[{result}]"
    else:
        result = result.split('<|im_start|>assistant')[1].split('<|im_end|>')[0]
        result = result.replace('`', '').replace('sql','')
    #cleaned = [line.strip().strip('"').strip("'") for line in lines]
    # cleaned = [re.sub(r'^\d+\.\s*', '', line).strip().strip('"') for line in lines]
    #print(lines)
    #print(cleaned)
    print(result)
    return result

def generate_data_model_llama(model,tokenizer,prompt,prompt_mode):
    
    messages = [
{"role": "user", "content": prompt},
]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    outputs = model.generate(**inputs, max_new_tokens=10000)
    output = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:])
    output = output.replace('<|eot_id|>','')
    print(output)
    if prompt_mode.lower() == 'rag':
        if isinstance(output, list):
            return output
        else:
            items = re.split(r"[,\.\-\*\•]", output)
            items = [item.replace('\n','').replace('*','').replace('#','').replace('/','').strip() for item in items if item.strip()]
            print(items)
            return items
    else:
        output = output.replace("`", "").replace("sql", "")
        return output

def generate_data_model_gemma(model,tokenizer,prompt,prompt_mode):
    messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": prompt}
        ]
    },
]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    outputs = model.generate(**inputs, max_new_tokens=10000)
    output = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:])
    if prompt_mode.lower() =='rag':
        output = output.replace('<end_of_turn>','')
    else:
        output = output.replace('<end_of_turn>','')
        output = output.replace('`','').replace('sql','')
    return output

if __name__== "__main__":
    model_choice = 'gemma'
    if model_choice.lower() == 'qwen':
        inference_model = "Qwen/Qwen3-4B-Instruct-2507"
    elif model_choice.lower() == 'hyper':
        inference_model = "naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B"
    elif model_choice.lower() == 'llama':
        inference_model = 'meta-llama/Llama-3.1-8B-Instruct'
    elif model_choice.lower() == 'gemma':
        inference_model = 'google/gemma-3-4b-it'
        
    model, tokenizer = model_load(inference_model)
    embedding_model = 'Qwen/Qwen3-Embedding-0.6B'
    chroma_path = "/home/ljm/web_modeler/Rag/chroma_db"
    collection_name = "pdf_chunks"
    business_desc = """
    오디오별로 트림을 설정하고 트림별 기본/선택 옵션을 셋팅하여 판매 스펙을 구성하는 업무입니다.
    각 트림에는 고유한 ID가 있고, 여러 옵션을 선택할 수 있습니다.
    옵션에는 패키지 옵션과 단품 옵션이 있으며, 각 옵션마다 가격이 책정됩니다.
    트림별로 적용 가능한 옵션 조건이 다르며, 이를 관리해야 합니다.
    """
    tables = [
    {"table_name": "car_trim", "description": "차량 트림(모델의 세부 사양) 정보를 저장"},
    {"table_name": "option", "description": "개별 옵션(예: 선루프, 가죽 시트 등) 정보를 저장"},
    {"table_name": "option_package", "description": "여러 옵션을 묶은 패키지 정보를 저장"},
    {"table_name": "trim_option_mapping", "description": "트림과 옵션 간의 매핑 관계를 저장 (N:M 관계)"}
   ]
    columns = [
    # car_trim
    {"table_name": "car_trim", "column_name": "trim_id", "data_type": "INT", "description": "트림의 고유 ID"},
    {"table_name": "car_trim", "column_name": "trim_name", "data_type": "VARCHAR(100)", "description": "트림 이름 (예: Luxury, Standard 등)"},
    {"table_name": "car_trim", "column_name": "base_price", "data_type": "DECIMAL(10,2)", "description": "트림의 기본 가격"},
    {"table_name": "car_trim", "column_name": "description", "data_type": "TEXT", "description": "트림 상세 설명"},
    {"table_name": "car_trim", "column_name": "created_at", "data_type": "DATETIME", "description": "등록 일시"},
    {"table_name": "car_trim", "column_name": "updated_at", "data_type": "DATETIME", "description": "수정 일시"}]
    
    prompt_mode = 'rag'
    query = need_question(model_choice, prompt_mode, business_desc,tables,columns)
    
    if isinstance(query, str):
        try:
            query = ast.literal_eval(query)
        except Exception:
            pass
        
    print(query)
    print(len(query))
    top_k=1
    doc_list = []
    # keyword 리스트 하나하나 관련문서 retrieve 
    for i in range(len(query)):
        doc = query_chroma(query[i],embedding_model,chroma_path,collection_name,top_k)
        doc_list.append(doc)
        
    prompt = build_prompt(tables,columns,business_desc, doc_list)
    prompt_mode = 'generate'
    if model_choice.lower() == 'qwen':
        result = generate_data_model_qwen(model,tokenizer,prompt,prompt_mode)
    elif model_choice.lower() == 'hyper':
        result = generate_data_model_hyper(model,tokenizer,prompt,prompt_mode)
    elif model_choice.lower() == 'llama':
        result = generate_data_model_llama(model,tokenizer,prompt,prompt_mode)    
    elif model_choice.lower() == 'gemma':
        result = generate_data_model_gemma(model,tokenizer,prompt,prompt_mode)  
    print('---------------------')
    print(result)
    print('---------------------')
