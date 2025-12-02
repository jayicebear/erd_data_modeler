from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import sys
sys.path.append('/home/ljm/web_modeler')
from main.column_guess import infer_column_meaning
app = FastAPI()

class ColumnRequest(BaseModel):
    input_column: str    

class ColumnResponse(BaseModel):
    column_meaning: str

@app.post("/column_guess", response_model=ColumnResponse)
def convert(request: ColumnRequest):
    try:
        column_guess = infer_column_meaning(
            column_name=request.input_column
        )
        return ColumnResponse(column_meaning=column_guess)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="192.168.0.40", port=8000)
