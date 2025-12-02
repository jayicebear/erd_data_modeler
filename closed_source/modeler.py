"""
자동화 데이터 거버넌스 구축 시스템
업무설명, 테이블, 컬럼을 입력받아 관련 테이블/컬럼을 유추하고 ERD 모델링을 생성
"""
import sys
sys.path.append('/home/ljm/web_modeler')

from openai import OpenAI
import json
from typing import List, Dict, Optional
import os
from dataclasses import dataclass, asdict
from Rag.retrieve import query_chroma
# OpenAI 클라이언트 초기화
client = OpenAI(api_key="")


@dataclass
class TableMetadata:
    """테이블 메타데이터"""
    table_name: str
    entity_name: str
    description: str
    columns: List[str]


@dataclass
class ColumnMetadata:
    """컬럼 메타데이터"""
    column_name: str
    data_type: str
    description: str
    table_name: str


@dataclass
class RelationshipMetadata:
    """테이블 간 관계 메타데이터"""
    from_table: str
    to_table: str
    from_column: str
    to_column: str
    relationship_type: str  # "1:1", "1:N", "N:M"

# =============================================================================
# 메인 함수
# =============================================================================

def format_tables_metadata(tables: List[TableMetadata]) -> str:
    """테이블 메타데이터를 문자열로 포맷팅"""
    result = []
    for i, table in enumerate(tables, 1):
        result.append(f"{i}. 테이블명: {table.table_name}")
        result.append(f"   엔티티명: {table.entity_name}")
        result.append(f"   설명: {table.description}")
        result.append(f"   컬럼: {', '.join(table.columns[:5])}...")  # 처음 5개만
        result.append("")
    return "\n".join(result)


def format_columns_metadata(columns: List[ColumnMetadata]) -> str:
    """컬럼 메타데이터를 문자열로 포맷팅"""
    result = []
    for i, col in enumerate(columns[:30], 1):  # 처음 30개만 표시
        result.append(f"{i}. {col.column_name} ({col.data_type}) - {col.description} [{col.table_name}]")
    result.append(f"... (총 {len(columns)}개 컬럼)")
    return "\n".join(result)

# Retrieve 가 필요한 부분을 찾아내는 부분 
def need_question(business_description: str,TABLES_METADATA: str, COLUMNS_METADATA: str) -> Dict:
    """
    데이터 모델링 생성에 필요한 자료를 유추하는 작업 
    """
    
    PROMPT = f"""
    당신은 데이터 모델링 및 데이터 거버넌스 전문가입니다.  
    아래의 **테이블 메타데이터**, **컬럼 메타데이터**, **업무 설명**을 분석하여  
    현재 모델링에 필요한 정보 중 **부족하거나 추가로 알아야 할 개념/데이터/업무요소**를 찾아주세요.  

    이 단계는 RAG 검색(Query Expansion)에 사용할 키워드를 생성하기 위한 것입니다.  
    
    ---
    ### 테이블 메타데이터
    {TABLES_METADATA}

    ### 컬럼 메타데이터
    {COLUMNS_METADATA}

    ### 업무 설명
    {business_description}

    ### 출력 형식
    - 데이터 모델링에 필요한 추가 정보나 개념을 **짧은 단어 또는 구 형태**로만 출력하세요.
    - 출력은 반드시 **리스트(list)** 형태로 작성하세요.
    - **불필요한 문장, 설명, 이유**는 포함하지 마세요.

    예시 출력:
    ["거래종류코드", "신용정보등급", "고객상세테이블", "거래분류프로세스", "데이터품질진단"]
    """

    print("=" * 80)
    print("Rag가 필요한 부분 search 중...")
    print("=" * 80)
        
    try:
        # GPT-4o-mini API 호출
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "당신은 정보 요구 추론 전문가입니다."
                },
                {
                    "role": "user",
                    "content": PROMPT
                }
            ],
            temperature=0.3,
            max_tokens=3000,
        )
        
        # 응답 파싱
        content = response.choices[0].message.content.strip()
        #print(content)
        clean_result = content.replace("```", "").replace('sql','').replace("\n", "").strip()
        print('----------------------------------')
        print('Quesitons:')
        print(clean_result)
        print('----------------------------------')
        return clean_result
        
    except json.JSONDecodeError as e:
        print(f"JSON 파싱 오류: {e}")
        print(f"응답 내용:\n{content}")
        return None
    except Exception as e:
        print(f"API 호출 오류: {e}")
        return None

def infer_data_model(business_description: str,TABLES_METADATA: str, COLUMNS_METADATA: str, doc) -> Dict:
    """
    업무 설명을 입력받아 관련 테이블과 컬럼을 추천하고 데이터 모델링 생성
    """
    
    INFERENCE_PROMPT = f"""
당신은 데이터 거버넌스 전문가입니다.
업무 설명을 분석하여 관련 테이블과 컬럼을 추천하고, 데이터 모델링을 생성하세요.
회사 내부 자료는 모델링시 참고하세요.
## 사용 가능한 테이블 메타데이터:
{TABLES_METADATA}

## 사용 가능한 컬럼 메타데이터:
{COLUMNS_METADATA}

## 업무 설명:
{business_description}

## 회사 내부 자료:
{doc}

## 요구사항:
1. 사용가능한 테이블, 사용가능한 컬럼, 업무설명을 보고 
SQL 로 데이터 모델링을 하세요 
** SQL 형식으로만 출력하세요. 설명은 포함하지 마세요.**
"""

    print("=" * 80)
    print("업무 설명 분석 중...")
    print("=" * 80)
    
    try:
        # GPT-4o-mini API 호출
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "당신은 데이터 모델링 전문가입니다."
                },
                {
                    "role": "user",
                    "content": INFERENCE_PROMPT
                }
            ],
            temperature=0.3,
            max_tokens=3000,
        )
        
        # 응답 파싱
        content = response.choices[0].message.content.strip()
        #print(content)
        clean_result = content.replace("```", "").replace('sql','').replace("\n", "").strip()
        print(clean_result)
        return clean_result
        
    except json.JSONDecodeError as e:
        print(f"JSON 파싱 오류: {e}")
        print(f"응답 내용:\n{content}")
        return None
    except Exception as e:
        print(f"API 호출 오류: {e}")
        return None


# =============================================================================
# 실행 
# =============================================================================

if __name__ == "__main__":
    model_name = 'Qwen/Qwen3-Embedding-0.6B'
    chroma_path = "/home/ljm/web_modeler/Rag/chroma_db"
    collection_name = "pdf_chunks"
    query = "거래종료고객에 대해 말해주세요"
    top_k=5
    doc = query_chroma(query,model_name,chroma_path,collection_name,top_k)
    print(doc)
    # 예시 1: 트림 설정 업무
    print("예시 1: 트림 설정 및 옵션 관리 업무")
    
    business_desc_1 = """
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
    
    questions =  need_question(business_desc_1,tables,columns)
    result_1 = infer_data_model(business_desc_1,tables,columns,doc)
    
    if result_1:
        # 결과 저장
        with open("result_trim_management.json", "w", encoding="utf-8") as f:
            json.dump(result_1, f, ensure_ascii=False, indent=2)
        
        print("\n결과 파일 저장 완료:")
    
    
