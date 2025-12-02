from sentence_transformers import SentenceTransformer
import chromadb

def query_chroma(query_text, model_name, chroma_path, collection_name, top_k=5):
    """
    ChromaDB에서 코사인 유사도로 쿼리 검색하는 함수

    Args:
        query_text (str): 검색할 질의문
        model_name (str): SentenceTransformer 임베딩 모델 이름
        chroma_path (str): ChromaDB 저장 경로
        collection_name (str): 컬렉션 이름
        top_k (int): 가져올 상위 문서 개수

    Returns:
        list: [(score, document)] 리스트
    """

    # 임베딩 모델 로드
    embedder = SentenceTransformer(model_name)

    # PersistentClient 로컬 DB 연결
    client = chromadb.PersistentClient(path=chroma_path)
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}  
    )
    
    #  쿼리 문장 임베딩
    query_embedding = embedder.encode(query_text).tolist()

    # Chroma에서 유사도 기반 검색
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "distances"]
    )

    #  결과 정리
    hits = []
    for doc, dist in zip(results["documents"][0], results["distances"][0]):
        hits.append({
            "score": 1 - dist,  # cosine similarity (Chroma는 distance이므로 1-변환)
            "content": doc
        })

    print(f" '{query_text}' 관련 상위 {top_k}개 문서:")
    for i, hit in enumerate(hits, 1):
        print(f"\n{i}. 유사도: {hit['score']:.4f}\n{hit['content'][:400]}...")

    return hits


if __name__ == "__main__":
    model_name = 'Qwen/Qwen3-Embedding-0.6B'
    chroma_path = "/home/ljm/web_modeler/Rag/chroma_db"
    collection_name = "pdf_chunks"

    query = "거래종료고객에 대해 말해주세요"
    query_chroma(query, model_name, chroma_path, collection_name, top_k=5)
