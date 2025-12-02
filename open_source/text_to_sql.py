from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import re 

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  

from transformers import AutoModelForCausalLM, AutoTokenizer


def model_load(model_name):
    # load the tokenizer and the model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    return model, tokenizer 

def build_prompt(schama, question):


    prompt = f"""
    You are an expert SQL developer.
    Convert the following natural language question into a valid SQL query.
    Use only the columns and tables listed below.

    Database schema:
    {schema}

    Question:
    {question}

    Return only the SQL code (no explanation).
    """
    
    return prompt

def sql_inference(model, tokenizer, prompt):
    messages = [
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    print('-------------------------')
    print(text)
    print('-------------------------')
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # conduct text completion
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=15000
    )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

    content = tokenizer.decode(output_ids, skip_special_tokens=True)
    content = content.replace('content:','')

    print("content:", content)

def sql_inference_hyper(prompt,model,tokenizer):
    chat = [
    {"role": "tool_list", "content": ""},
    {"role": "system", "content": "You are an expert data model analyst."},
    {"role": "user", "content": prompt},
]

    inputs = tokenizer.apply_chat_template(chat, add_generation_prompt=True, return_dict=True, return_tensors="pt")
    inputs = inputs.to("cuda")
    output_ids = model.generate(
        **inputs,
        max_length=1024,
        stop_strings=["<|endofturn|>", "<|stop|>"],
        tokenizer=tokenizer
        )
    result = tokenizer.batch_decode(output_ids)[0]
    result = result.split('<|im_start|>assistant')[1].split('<|im_end|>')[0]
    lines = [line.strip() for line in result.strip().splitlines() if line.strip()]
    cleaned = [re.sub(r'^\d+\.\s*', '', line).strip().strip('"') for line in lines]
    print(lines)
    print(cleaned)
    return cleaned

if __name__ =='__main__':
    model_name = "Qwen/Qwen3-4B-Instruct-2507"
    model_name = "naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B"

    model, tokenizer = model_load(model_name)
    schema = """
    Table: customers(id, name, age, city)
    Table: orders(id, customer_id, amount, date)
    """
    question = "서울에 사는 고객들의 평균 주문 금액을 보여줘"
    # prepare the model input
    prompt = build_prompt(schema,question)
    sql = sql_inference(model, tokenizer, prompt)
    
    print(sql)
