from openai import OpenAI
import json
from pathlib import Path
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language, MarkdownHeaderTextSplitter
import os
import chromadb
from sentence_transformers import SentenceTransformer

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  

def load_and_chunk_pdf(pdf_path, chunk_size=1000, chunk_overlap=100):
    """
    PDF를 마크다운으로 변환 후 청킹하여 시각화
    
    Args:
        pdf_path (str): PDF 파일 경로
        chunk_size (int): 청크 크기
        chunk_overlap (int): 청크 오버랩
    
    Returns:
        list: 청크 리스트
    """
    # docling 으로 PDF-> 마크다운으로 변환
    pipeline_options = PdfPipelineOptions()
    pipeline_options.generate_picture_images = True
    
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )
    
    result = converter.convert(pdf_path)
    full_text = result.document.export_to_markdown(image_mode="referenced")
    
    # 헤더청킹
    # 문서를 분할할 헤더 레벨 & 이름 정의
    headers_to_split_on = [  
    (
        "#",
        "Title",
    ),  
    (
        "##",
        "Section",
    ),  
    (
        "\n\n",
        "Subsection",
    ),  
    ]
    # 헤더 spliter 정의
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on, strip_headers=False)
    # 청크 만들기 
    chunks = markdown_splitter.split_text(full_text)
    
    return chunks

def save_chunks_to_json(chunks, output_path):
    """
    Document 청크 데이터를 JSON 파일로 저장하는 함수

    Args:
        chunks (list): Document 객체 리스트
        output_path (str): 저장할 JSON 파일 경로
    """
    # Document 객체를 dict 형태로 변환
    chunk_data = []
    for i, chunk in enumerate(chunks):
        chunk_data.append({
            "id": i + 1,
            "content": getattr(chunk, "page_content", str(chunk)),  # 텍스트만 추출
        })

    # 디렉터리가 없으면 생성
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # JSON 파일 저장
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunk_data, f, ensure_ascii=False, indent=2)

    print(f" {len(chunk_data)}개 청크를 '{output_path}'에 저장했습니다.")


def save_chunks_to_chroma(chunks, model_name, chroma_path, collection_name):
    """청크 데이터를 바로 ChromaDB에 저장"""
    embedder = SentenceTransformer(model_name)
    client = chromadb.PersistentClient(path=chroma_path)
    collection = client.get_or_create_collection(name=collection_name,metadata={"hnsw:space": "cosine"})

    for i, chunk in enumerate(chunks):
        content = getattr(chunk, "page_content", str(chunk))
        embedding = embedder.encode(content).tolist()
        collection.add(
            ids=[str(i + 1)],
            embeddings=[embedding],
            documents=[content],
        )

    print(f"{len(chunks)}개의 청크를 '{collection_name}' 컬렉션에 직접 저장했습니다.")
    
if __name__ == "__main__":
    file_name = '산업은행_교육자료'
    pdf_path = f'/home/ljm/web_modeler/pdf/{file_name}.pdf'
    output_path = f'/home/ljm/web_modeler/Rag/chunked_data/{file_name}.json'
    model_name = 'Qwen/Qwen3-Embedding-0.6B'
    chroma_path = "/home/ljm/web_modeler/Rag/chroma_db"
    collection_name ="pdf_chunks"
    chunk = load_and_chunk_pdf(pdf_path)
    # 청크 json 으로 저장
    #save_chunks_to_json(chunk, output_path)
    # 청크 chroma에 저장
    save_chunks_to_chroma(chunk,model_name,chroma_path, collection_name)
