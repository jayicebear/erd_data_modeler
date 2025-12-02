
from openai import OpenAI

client = OpenAI(api_key='')
                
# 프롬프트
PROMPT_TEMPLATE = """
You are a data schema understanding assistant.
Given a column name from a database, infer the most likely human-readable meaning or expanded form.
Return 3 ranked candidate interpretations.

Examples:
- "cust_id" → ["customer_id", "customer identifier", "고객 식별자"]
- "txn_amt" → ["transaction_amount", "거래 금액", "결제액"]
- "emp_dept_cd" → ["employee_department_code", "사원 부서 코드", "직원 부서 식별자"]

Now analyze the following column name and suggest the 3 most likely meanings.
Column: {column_name}
"""

def infer_column_meaning(column_name: str):
    """컬럼명을 입력하면 GPT-4o mini가 의미 후보 3개를 반환"""
    prompt = PROMPT_TEMPLATE.format(column_name=column_name)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an expert data model analyst."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.4,   # 낮을수록 안정된 결과
        max_tokens=150,
    )
    result = response.choices[0].message.content
    result = result.strip("`").replace("json", "").strip()
    return result

# 테스트
if __name__ == "__main__":
    column = "cstmer_txsx_034"
    result = infer_column_meaning(column)
    print(result)
