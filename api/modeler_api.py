from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
from openai import OpenAI
import json
import os
import sys 
sys.path.append('/home/ljm/web_modeler')
from main.modeler import infer_data_model
# FastAPI 앱 초기화
app = FastAPI(title="Data Modeler API", version="1.0.0")

# =============================================================================
# Request/Response 모델
# =============================================================================

class ModelingRequest(BaseModel):
    business_description: str
    COLUMNS_METADATA: str 
    TABLES_METADATA: str 
    
    
# class TableRecommendation(BaseModel):
#     table_name: str
#     entity_name: str
#     description: str
#     reason: str
#     columns: List[str]

# class Relationship(BaseModel):
#     from_table: str
#     to_table: str
#     from_column: str
#     to_column: str
#     relationship_type: str
#     description: str

class ModelingResponse(BaseModel):
    #business_rules: List[str]
    #recommended_tables: List[TableRecommendation] 
    erd: str  
    #relationships: List[Relationship] 
# =============================================================================
# API 엔드포인트
# =============================================================================


@app.post("/modeler", response_model=ModelingResponse)
def create_data_model(request: ModelingRequest):
    """
    업무 설명을 받아 데이터 모델 추천
    """
    try:
        # LLM 호출
        result = infer_data_model(request.business_description,request.TABLES_METADATA,request.COLUMNS_METADATA)
        
        # Mermaid & SQL 생성
        # result["mermaid_code"] = generate_erd_mermaid(result)
        # result["sql_ddl"] = generate_sql_ddl(result)
        if isinstance(result, str):
            return {"erd": result}
        else:
            return result
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail=f"JSON 파싱 오류: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"오류 발생: {str(e)}")

# =============================================================================
# 실행
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="192.168.0.40", port=8000)
