# ERD(entity relationship diagram) Building LLM Agent
**"업무 한 줄 설명"만으로 테이블/컬럼 유추 → ERD 자동 생성**!

기업 내 수백 개 도메인에 흩어진 데이터 모델을 빠르게 정리하고 싶을 때,  
데이터 거버넌스 팀이 매번 직접 ERD를 그리는 수고를 덜어줍니다.

## 핵심 기능

| 모듈                | 기능                                                                 |
|---------------------|----------------------------------------------------------------------|
| `column_guess.py` | 컬럼명만 보고 의미 추측 (예: `cust_no` → "고객번호", `ord_amt` → "주문금액") |
| `text_to_sql.py`    | 자연어 질문 → SQL 자동 변환 ( Text-to-SQL)                   |
| `modeler.py`        | 업무설명 + 기존 테이블/컬럼 → 연관 테이블 자동 유추 → ERD 생성            |
| `RAG + ChromaDB`      | 과거 데이터 모델링 사례, 내부문서, 모델링 가이드등을 벡터 DB에 저장 → column_guess,modeler 구축 시 활용          |
| `API(Fast api)` | 생성된 ERD(Mermaid) Fast api 활용하여 필요한 부분으로 전송            |

## 사용 모델

| 모델명                                      | 파라미터 | 특징 / 용도                                      |
|--------------------------------------------|----------|--------------------------------------------------|
| `Qwen/Qwen3-4B-Instruct-2507`              | 4B       | 한국어 성능 우수, 컬럼명 추측 및 긴 문맥 이해에 최적 |
| `naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B` | 1.5B     | 경량 + 한국어 특화, 로컬/엣지 환경에서 빠른 추론       |
| `meta-llama/Llama-3.1-8B-Instruct`         | 8B       | 범용적, 복잡한 ERD 관계 유추 및 Text-to-SQL 정확도 ↑ |
| `google/gemma-3-4b-it`                     | 4B       | 가볍고 빠른 추론, 비용 효율 최고                      |

> **기본 전략**: 컬럼명 추측 → HyperCLOVAX 1.5B / 관계 유추 & ERD 생성 → Llama-3.1-8B (성능 우선 시)
