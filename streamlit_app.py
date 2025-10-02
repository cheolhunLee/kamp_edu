## streamlit 관련 모듈 불러오기
import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile

from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings

from langchain_core.documents.base import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_core.runnables import Runnable
from langchain.schema.output_parser import StrOutputParser
from langchain_community.document_loaders import PyMuPDFLoader
from typing import List
import os
import fitz  # PyMuPDF
import re

############################### 1단계 : PDF 문서를 벡터DB에 저장하는 함수들 ##########################

## 1: 임시폴더에 파일 저장
def save_uploadedfile(uploadedfile: UploadedFile) -> str : 
    temp_dir = "PDF_임시폴더"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    file_path = os.path.join(temp_dir, uploadedfile.name)
    with open(file_path, "wb") as f:
        f.write(uploadedfile.read()) 
    return file_path

## 2: 저장된 PDF 파일을 Document로 변환
def pdf_to_documents(pdf_path:str) -> List[Document]:
    documents = []
    loader = PyMuPDFLoader(pdf_path)
    doc = loader.load()
    for d in doc:
        d.metadata['file_path'] = pdf_path
    documents.extend(doc)
    return documents

## 3: Document를 더 작은 document로 변환
def chunk_documents(documents: List[Document]) -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    return text_splitter.split_documents(documents)

## 4: Document를 벡터DB로 저장
@st.cache_resource(show_spinner=False)
def get_embeddings_model():
    return OllamaEmbeddings(model="exaone3.5:7.8b")

def save_to_vector_store(documents: List[Document], progress_bar=None, status_text=None) -> None:
    embeddings = get_embeddings_model()
    
    if progress_bar and status_text:
        status_text.text("임베딩 벡터 생성 중...")
        
        # 페이지별로 그룹화
        page_groups = {}
        for doc in documents:
            page_num = doc.metadata.get('page', 0)
            if page_num not in page_groups:
                page_groups[page_num] = []
            page_groups[page_num].append(doc)
        
        total_pages = len(page_groups)
        processed_pages = 0
        
        # 페이지별로 임베딩 생성
        all_docs = []
        for page_num in sorted(page_groups.keys()):
            page_docs = page_groups[page_num]
            
            # 해당 페이지의 모든 청크 임베딩
            for doc in page_docs:
                temp_vector_store = FAISS.from_documents([doc], embedding=embeddings)
                all_docs.append(doc)
            
            processed_pages += 1
            
            # 진행률 업데이트 (페이지 단위)
            progress = processed_pages / total_pages
            progress_bar.progress(0.3 + (progress * 0.6))  # 30%~90% 범위에서 진행
            status_text.text(f"임베딩 진행 중... {processed_pages}/{total_pages} 페이지 처리 완료 ({progress*100:.1f}%)")
        
        # 최종 벡터 스토어 생성
        status_text.text("벡터 데이터베이스 저장 중...")
        vector_store = FAISS.from_documents(documents, embedding=embeddings)
    else:
        # 기존 방식 (진행률 표시 없음)
        vector_store = FAISS.from_documents(documents, embedding=embeddings)
    
    vector_store.save_local("faiss_index")


############################### 2단계 : RAG 기능 구현과 관련된 함수들 ##########################

## 사용자 질문에 대한 RAG 처리
@st.cache_data
def process_question(user_question):
    embeddings = get_embeddings_model()
    ## 벡터 DB 호출
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

    ## 관련 문서 3개를 호출하는 Retriever 생성
    retriever = new_db.as_retriever(search_kwargs={"k": 3})
    ## 사용자 질문을 기반으로 관련문서 3개 검색 
    retrieve_docs : List[Document] = retriever.invoke(user_question)

    ## RAG 체인 선언
    chain = get_rag_chain()
    ## 질문과 문맥을 넣어서 체인 결과 호출
    response = chain.invoke({"question": user_question, "context": retrieve_docs})
    return response, retrieve_docs
    
# Exaone 생성 모델 사용
@st.cache_resource
def get_exaone_model():
    return Ollama(model="exaone3.5:7.8b", temperature=0)

def get_rag_chain() -> Runnable:
    template = """
    다음의 컨텍스트를 활용해서 질문에 답변해줘
    - 질문에 대한 응답을 해줘
    - 간결하게 5줄 이내로 해줘
    - 곧바로 응답결과를 말해줘
    컨텍스트 : {context}
    질문: {question}
    응답:"""
    custom_rag_prompt = PromptTemplate.from_template(template)
    model = get_exaone_model()
    return custom_rag_prompt | model | StrOutputParser()

############################### 3단계 : 응답결과와 문서를 함께 보도록 도와주는 함수 ##########################
@st.cache_data(show_spinner=False)
def convert_pdf_to_images(pdf_path: str, dpi: int = 250) -> List[str]:
    doc = fitz.open(pdf_path)  # 문서 열기
    image_paths = []
    
    # 이미지 저장용 폴더 생성
    output_folder = "PDF_이미지"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for page_num in range(len(doc)):  #  각 페이지를 순회
        page = doc.load_page(page_num)  # 페이지 로드

        zoom = dpi / 72  # 72이 디폴트 DPI
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat) # type: ignore

        image_path = os.path.join(output_folder, f"page_{page_num + 1}.png")  # 페이지 이미지 저장 page_1.png, page_2.png, etc.
        pix.save(image_path)  # PNG 형태로 저장
        image_paths.append(image_path)  # 경로를 저장
        
    return image_paths

def display_pdf_page(image_path: str, page_number: int) -> None:
    image_bytes = open(image_path, "rb").read()  # 파일에서 이미지 인식
    st.image(image_bytes, caption=f"Page {page_number}", output_format="PNG", width=600)


def natural_sort_key(s):
    return [int(text) if text.isdigit() else text for text in re.split(r'(\d+)', s)]

def main():
    st.set_page_config("PDF FAQ 챗봇", layout="wide")

    left_column, right_column = st.columns([1, 1])
    with left_column:
        st.header("PDF FAQ 챗봇")

        pdf_doc = st.file_uploader("PDF Uploader", type="pdf")
        button =  st.button("PDF 업로드하기")
        if pdf_doc and button:
            # 진행률 표시용 컴포넌트 생성
            progress_bar = st.progress(0)
            status_text = st.empty()
            try:
                # 1단계: PDF 저장 및 문서 변환
                status_text.text("PDF 파일 저장 중...")
                progress_bar.progress(0.1)
                pdf_path = save_uploadedfile(pdf_doc)
                
                status_text.text(" PDF 문서 로딩 중...")
                progress_bar.progress(0.2)
                pdf_document = pdf_to_documents(pdf_path)
                
                status_text.text(" 문서 청킹 중...")
                progress_bar.progress(0.3)
                smaller_documents = chunk_documents(pdf_document)
                
                # 2단계: 벡터 저장
                save_to_vector_store(smaller_documents, progress_bar, status_text)
                
                # 3단계: PDF를 이미지로 변환
                status_text.text("PDF 페이지를 이미지로 변환 중...")
                progress_bar.progress(0.95)
                images = convert_pdf_to_images(pdf_path)
                st.session_state.images = images
                
                # 완료
                progress_bar.progress(1.0)
                status_text.text(" 업로드 완료!")
                
                # 성공 메시지 표시
                st.success(f"PDF 업로드가 완료되었습니다! (총 {len(pdf_document)}페이지)")
                
            except Exception as e:
                status_text.text(f" 오류 발생: {str(e)}")
                st.error(f"업로드 중 오류가 발생했습니다: {str(e)}")
            
            finally:
                # 3초 후 진행률 표시 제거
                if st.button("진행률 숨기기") or True:  # 자동으로 숨기기
                    pass

        user_question = st.text_input("PDF 문서에 대해서 질문해 주세요",placeholder="궁금하신 내용을 적어주세요")

        if user_question:
            response, context = process_question(user_question)
            st.write(response)
            i = 0 
            for document in context:
                with st.expander("관련 문서"):
                    st.write(document.page_content)
                    file_path = document.metadata.get('source', '')
                    page_number = document.metadata.get('page', 0) + 1
                    button_key =f"link_{file_path}_{page_number}_{i}"
                    reference_button = st.button(f" {os.path.basename(file_path)} pg.{page_number}", key=button_key)
                    if reference_button:
                        st.session_state.page_number = str(page_number)
                    i = i + 1
    with right_column:
        # page_number 호출
        page_number = st.session_state.get('page_number')
        if page_number:
            page_number = int(page_number)
            image_folder = "PDF_이미지"
            images = sorted(os.listdir(image_folder), key=natural_sort_key)
            image_paths = [os.path.join(image_folder, image) for image in images]
            print(image_paths[page_number - 1])
            display_pdf_page(image_paths[page_number - 1], page_number)


if __name__ == "__main__":
    main()
