from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import sys
sys.path.append('/home/ljm/web_modeler')
from main.text_to_sql import text_to_sql
app = FastAPI()

class SQLRequest(BaseModel):
    schema_description: str
    question: str

class SQLResponse(BaseModel):
    sql_query: str

@app.post("/text-to-sql", response_model=SQLResponse)
def convert(request: SQLRequest):
    try:
        sql = text_to_sql(
            schema_description=request.schema_description,
            question=request.question
        )
        return SQLResponse(sql_query=sql)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="192.168.0.40", port=8000)
