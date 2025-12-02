from openai import OpenAI

client = OpenAI(api_key="")

PROMPT = """
You are an expert SQL developer.
Convert the following natural language question into a valid SQL query.
Use only the columns and tables listed below.

Database schema:
{schema_description}

Question:
{user_question}

Return only the SQL code (no explanation).
"""

def text_to_sql(schema_description: str, question: str):
    prompt = PROMPT.format(
        schema_description=schema_description,
        user_question=question
    )
    print(prompt)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a SQL expert."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
        max_tokens=200,
    )
    answer = response.choices[0].message.content.strip()
    result = answer.strip("`").replace("sql", "").strip()
    return result

# 예시 실행
if __name__ == "__main__":
    schema = """
    Table: customers(id, name, age, city)
    Table: orders(id, customer_id, amount, date)
    """
    question = "서울에 사는 고객들의 평균 주문 금액을 보여줘"
    sql_query = text_to_sql(schema, question)
    result = sql_query.strip("`").replace("sql", "").strip()
    print(result)
