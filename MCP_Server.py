from mcp.server.fastmcp import FastMCP
from sentence_transformers import SentenceTransformer
import fitz
import os

# LangChain 관련 추가
from langchain.chains import RetrievalQA
from langchain.llms import Ollama
from langchain.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings

# MCP 서버
mcp = FastMCP(
    "DocumentQAServer",
    instructions="문서를 업로드하고, 질문하면 문서 기반 요약 응답을 제공합니다.",
    host="0.0.0.0",
    port=8005,
)

# 문서 임베딩 및 ChromaDB 설정
persist_dir = "./chroma_db"
os.makedirs(persist_dir, exist_ok=True)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
embedding = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
vectordb = Chroma(persist_directory=persist_dir, embedding_function=embedding)

# 문서 업로드 함수
def read_pdf_text(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

@mcp.tool()
async def upload_document(file_path: str) -> str:
    if not os.path.exists(file_path):
        return f"[ERROR] 파일이 없습니다: {file_path}"

    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        text = read_pdf_text(file_path)
    elif ext in [".txt"]:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
    else:
        return f"[ERROR] 지원하지 않는 파일 형식: {ext}"

    embedding_vector = embedding_model.encode([text])[0]
    vectordb.add_texts([text], embeddings=[embedding_vector.tolist()], ids=[os.path.basename(file_path)])
    
    return {"status": "문서 저장 완료"}

# LangChain QA 체인 설정
llm = Ollama(model="exaone3.5:7.8b")
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectordb.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=False
)

# 문서 기반 질문 응답 함수
@mcp.tool()
async def ask_question(query: str) -> str:
    try:
        answer = qa_chain.run(query)
        return answer
    except Exception as e:
        return f"[ERROR] 질문 처리 중 오류 발생: {str( e)}"
    
@mcp.tool()
async def list_documents() -> str:
    """업로드된 문서 목록을 조회합니다."""
    try:
        # ChromaDB에서 저장된 문서 ID들 조회
        collection = vectordb._collection
        docs = collection.get()
        
        if not docs['ids']:
            return " 저장된 문서가 없습니다."
        
        # 문서 목록 정리
        doc_count = len(docs['ids'])
        doc_list = "\n".join([f"  📄 {i+1}. {doc_id}" for i, doc_id in enumerate(docs['ids'])])
        
        return f" 총 {doc_count}개의 문서가 저장되어 있습니다:\n\n{doc_list}"
        
    except Exception as e:
        return f"[ERROR] 문서 목록 조회 중 오류: {str(e)}"

# 서버 실행
if __name__ == "__main__":
    mcp.run(transport="sse")

