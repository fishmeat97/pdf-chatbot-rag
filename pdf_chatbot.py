import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
import os

# --- 1. 頁面設定 ---
st.set_page_config(page_title="My AI Research Assistant", page_icon="🤖")
st.header("🤖 Chat with your PDF (RAG Prototype)")

# --- 2. 側邊欄：設定與上傳 ---
with st.sidebar:
    st.title("Configuration")
    # 使用 password 類型隱藏 API Key
    openai_api_key = st.text_input("Enter OpenAI API Key", type="password")
    uploaded_file = st.file_uploader("Upload your PDF here", type="pdf")
    
    st.markdown("---")
    st.markdown("""
    **How it works:**
    1. **Ingest**: Reads the PDF text.
    2. **Split**: Breaks text into chunks.
    3. **Embed**: Converts text to numbers (Vectors).
    4. **Store**: Saves vectors in FAISS (Vector DB).
    5. **Retrieve**: Finds relevant info for your query.
    """)

# --- 3. 核心邏輯函數 ---

def get_pdf_text(pdf_docs):
    """讀取 PDF 文字"""
    text = ""
    pdf_reader = PdfReader(pdf_docs)
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text

def get_text_chunks(text):
    """將文字切分成小塊 (Chunks)"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks, api_key):
    """將文字轉為向量並存入資料庫 (FAISS)"""
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore, api_key):
    """建立對話鏈 (LangChain 的核心)"""
    llm = ChatOpenAI(openai_api_key=api_key, temperature=0.5, model_name="gpt-3.5-turbo")
    
    memory = ConversationBufferMemory(
        memory_key='chat_history', 
        return_messages=True
    )
    
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

# --- 4. 主程式邏輯 ---

if openai_api_key:
    # 初始化 session state
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    # 當使用者上傳檔案後
    if uploaded_file is not None:
        # 只有在還沒處理過檔案時才執行
        if st.session_state.conversation is None:
            with st.spinner("Processing PDF... (Extracting -> Chunking -> Embedding)"):
                try:
                    # A. 讀取 PDF
                    raw_text = get_pdf_text(uploaded_file)
                    
                    if not raw_text:
                        st.error("Could not extract text from this PDF. It might be scanned images.")
                    else:
                        # B. 切分文字
                        text_chunks = get_text_chunks(raw_text)
                        
                        # C. 建立向量資料庫
                        vectorstore = get_vectorstore(text_chunks, openai_api_key)
                        
                        # D. 建立對話鏈
                        st.session_state.conversation = get_conversation_chain(vectorstore, openai_api_key)
                        
                        st.success("PDF Processed! You can now ask questions.")
                except Exception as e:
                    st.error(f"An error occurred: {e}")

        # --- 5. 聊天介面 ---
        user_question = st.text_input("Ask a question about your document:")
        
        if user_question:
            if st.session_state.conversation:
                with st.spinner("Thinking..."):
                    response = st.session_state.conversation({'question': user_question})
                    st.session_state.chat_history = response['chat_history']

                # 顯示對話紀錄
                for i, message in enumerate(reversed(st.session_state.chat_history)):
                    if i % 2 == 0: # AI 的回答
                        st.markdown(f"🤖 **AI:** {message.content}")
                        st.markdown("---")
                    else: # 妳的問題
                        st.markdown(f"👤 **You:** {message.content}")
else:
    st.warning("Please enter your OpenAI API Key in the sidebar to start.")
