
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import re 

os.environ["CUDA_VISIBLE_DEVICES"] = "3"  

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

def build_prompt(column_name):
    
    prompt = f"""
    You are a data schema understanding assistant.
    Given a column name from a database, infer the most likely human-readable meaning or expanded form.
    Return 3 ranked candidate interpretations. Do not explain about the possible meaning. Just give me three possible name.

    Examples:
    - "cust_id" → ["customer_id", "customer identifier", "고객 식별자"]
    - "txn_amt" → ["transaction_amount", "거래 금액", "결제액"]
    - "emp_dept_cd" → ["employee_department_code", "사원 부서 코드", "직원 부서 식별자"]

    Now analyze the following column name and suggest the 3 most likely meanings.
    Column: {column_name}
    """
    return prompt

def column_inference(prompt,model,tokenizer):
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
    return content

def column_inference_hyper(prompt,model,tokenizer):
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
    column_name = "cstmer_txsx_034"
    model, tokenizer = model_load(model_name)
    prompt = build_prompt(column_name)
    result = column_inference(prompt,model,tokenizer)
    result_hyper = column_inference_hyper(prompt,model,tokenizer)

    print(result)
