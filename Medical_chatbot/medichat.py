import os
import streamlit as st
from streamlit.components.v1 import html
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
from dotenv import load_dotenv, find_dotenv

# Load environment variables
load_dotenv(find_dotenv())

DB_FAISS_PATH="vectorstore/db_faiss"

st.set_page_config(page_title="Medichat", layout="wide")

st.markdown('''
    <style>
        body {
            background-color: #0d1117;
            color: #c9d1d9;
        }
        .assistant {
            background-color: #161b22;
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 10px;
            color: #c9d1d9;
        }
        .user {
            background-color: #238636;
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 10px;
            text-align: right;
            color: #ffffff;
        }
        input {
            background-color: #0d1117;
            color: #c9d1d9;
            border: 1px solid #30363d;
        }
        .st-expander {
            background-color: #161b22;
            border-radius: 10px;
            padding: 10px;
            margin-top: 10px;
        }
    </style>
''', unsafe_allow_html=True)

@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

def set_custom_prompt(custom_prompt_template):
    return PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])

def load_llm(huggingface_repo_id, HF_TOKEN):
    llm = HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        task="text-generation",
        temperature=0.3,
        model_kwargs={"token": HF_TOKEN, "max_length": 1024}
    )
    return llm

def main():
    st.title("MediChat")

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        role, content = message['role'], message['content']
        if role == 'user':
            st.markdown(f'<div class="user">{content}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="assistant">{content}</div>', unsafe_allow_html=True)

    prompt = st.text_input("Ask something...")

    if prompt:
        st.markdown(f'<div class="user">{prompt}</div>', unsafe_allow_html=True)
        st.session_state.messages.append({'role': 'user', 'content': prompt})

        CUSTOM_PROMPT_TEMPLATE = '''
Use the pieces of information provided in the context to answer the user's question as accurately and comprehensively as possible.
If you don't know the answer, just say you don't know. Don't try to make up an answer.
Don't provide anything out of the given context.

Context: {context}
Question: {question}

Provide a clear, detailed answer.
'''

        HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
        HF_TOKEN = os.environ.get("HF_TOKEN")

        try:
            vectorstore = get_vectorstore()
            if vectorstore is None:
                st.error("Failed to load the vector store")

            qa_chain = RetrievalQA.from_chain_type(
                llm=load_llm(huggingface_repo_id=HUGGINGFACE_REPO_ID, HF_TOKEN=HF_TOKEN),
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={'k': 5}),
                return_source_documents=True,
                chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
            )

            response = qa_chain.invoke({'query': prompt})
            result = response["result"]
            source_documents = response["source_documents"]

            st.markdown(f'<div class="assistant">{result}</div>', unsafe_allow_html=True)
            st.session_state.messages.append({'role': 'assistant', 'content': result})

            st.subheader("Source Documents")
            for i, doc in enumerate(source_documents):
                with st.expander(f" Document {i+1}"):
                    st.markdown(f'<div class="assistant">{doc.page_content}</div>', unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
