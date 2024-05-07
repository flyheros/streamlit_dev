import streamlit as st
import tiktoken
from loguru import logger

from langchain.chains  import ConversationalRetrievalChain
# from langchain.chat_models import ChatOpenAi

from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import Docx2txtLoader
from langchain.document_loaders import UnstructuredPowerPointLoader

# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.embeddings import HuggingfaceEmbeddings

# from langchain.memory import ConversationBufferMemory
# from langchain.vectorstores import FAISS

# from langchain.callbacks import get_openai_callback
from langchain.memory import StreamlitChatMessageHistory


import pandas as pd

def main():
    st.set_page_config(
        page_title="DirChat",
        page_icon=":books:"
    )

    st.title("_Private Data :red[QA chat]_ :books:")
    
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
        
    with st.sidebar:
        upload_files = st.file_uploader("Upload your file", type="")
        openai_api_key = st.text_input("OPENAI API KEY", key="")
        process = st.button("Process")
        
    if process:
        if not openai_api_key:
            st.info("Please add your OpenAI API Key to continue")
            st.stop()
            
        files_text  = get_text(uploaded_files)
        text_chunks = get_text_chunks(files_text)
        vetorestore = get_vectorstore(text_chunks)
        
        st.session_state.conversation = get_conversation_chain(vectorestore, openai_api_key)

    if 'message' not in st.session_state:
        print('초기화')
        st.session_state['message'] = [{'role':'assistant'
                                        , 'content':'안녕하세요!주어진 문서에 대해 궁금하신게 있나요?'}]
    
    for message in st.session_state['message'] :
        with st.chat_message(message['role']):
            st.markdown(message['content'])
            
            

    history = StreamlitChatMessageHistory(key='chat_message')
    
    if query := st.chat_input("질문을 입력해"):
        st.session_state['message'].append({"role":"user", "content":query})
    
        with st.chat_message("user"):
            st.markdown(query)
            
        with st.chat_message("assistant"):
            chain = st.session_state.conversation
            
            with st.spinner("THINKING..."):
                result = chain({"question": query})
                with get_openai_callback() as cb:
                    st.session_state.chat_history = result['chat_history']
                response = result['answer']
                source_documents= result['source_documents']
                st.markdown(response)
                with st.expander('참고 문서 확인'):
                    st.markdown(source_documents[0].metadata['source'], help=source_documents[0].page_count)


main()
