import streamlit as st
import os
import dotenv
from time import time

from langchain_community.document_loaders.text import TextLoader
from langchain_community.document_loaders import (
    WebBaseLoader,
    PyPDFLoader,
    Docx2txtLoader,
)

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.llms import Ollama
from langchain_core.output_parsers import StrOutputParser

dotenv.load_dotenv()

DB_DOCS_LIMIT = 10

llm_stream = Ollama(
    model="deepseek-r1:8b",
    temperature=0.3,
)


def load_doc_to_db():
    # Use loader according to doc type
    if "rag_docs" in st.session_state and st.session_state.rag_docs:
        docs = []
        for doc_file in st.session_state.rag_docs:
            if doc_file.name not in st.session_state.rag_sources:
                if len(st.session_state.rag_sources) < DB_DOCS_LIMIT:
                    os.makedirs("source_files", exist_ok=True)
                    file_path = f"./source_files/{doc_file.name}"
                    with open(file_path, "wb") as file:
                        file.write(doc_file.read())

                    try:
                        if doc_file.type == "application/pdf":
                            loader = PyPDFLoader(str(file_path))
                        elif doc_file.endswith(".docx"):
                            loader = Docx2txtLoader(str(file_path))
                        elif doc_file.endswith(".txt"):
                            loader = TextLoader(str(file_path))
                        else:
                            st.warning(
                                f"Document type {doc_file.type} is not supported"
                            )
                            continue

                        docs.extend(loader.load())
                        st.session_state.rag_sources.append(doc_file.name)
                    except Exception as e:
                        st.toast(f"Error loading document: {e}")
                        print(f"Error loading document: {e}")
                    finally:
                        os.remove(file_path)
                else:
                    st.error(f"Maximum number of documents ({DB_DOCS_LIMIT}) reached")

        if docs:
            _split_and_load_docs(docs)
            st.toast(f"Documents loaded {len(docs)} successfully")


def load_url_to_db():
    if "rag_url" in st.session_state and st.session_state.rag_url:
        url = st.session_state.rag_url
        docs = []
        if url not in st.session_state.rag_sources:
            if len(st.session_state.rag_sources) < DB_DOCS_LIMIT:
                try:
                    loader = WebBaseLoader(url)
                    docs = loader.load()
                    st.session_state.rag_sources.append(url)
                except Exception as e:
                    st.toast(f"Error loading URL: {e}")
                    print(f"Error loading URL: {e}")

                if docs:
                    _split_and_load_docs(docs)
                    st.toast(f"URL loaded {len(docs)} successfully")
            else:
                st.error(f"Maximum number of documents ({DB_DOCS_LIMIT}) reached")


def _split_and_load_docs(docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=5000,
        chunk_overlap=1000,
    )
    document_chunks = text_splitter.split_documents(docs)

    if "vector_db" not in st.session_state:
        st.session_state.vector_db = _initialize_vector_db(docs)
    else:
        st.session_state.vector_db.add_documents(document_chunks)


def _initialize_vector_db(docs):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_db = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=f"{str(time()).replace('.', '')[:14]}_"
        + st.session_state["session_id"],
    )

    chroma_client = vector_db._client
    collection_names = sorted(
        [collection.name for collection in chroma_client.list_collections()]
    )
    print(collection_names)
    while len(collection_names) > 20:
        chroma_client.delete_collection(name=collection_names[0])
        collection_names.pop(0)

    return vector_db


# --- Retrieval Augmented Generation (RAG) Phase ---
def _get_context_retriever_chain(vector_db, llm):
    retriever = vector_db.as_retriever()

    prompt = ChatPromptTemplate.from_messages(
        [
            MessagesPlaceholder(variable_name="messages"),
            ("user", "{input}"),
            (
                "user",
                "Given the above conversation, generate a search query to look up in order to get inforamtion relevant to the conversation, focusing on the most recent messages.",
            ),
        ]
    )

    return create_history_aware_retriever(llm, retriever, prompt)


def _get_conversational_rag_chain(llm):
    retriever_chain = _get_context_retriever_chain(st.session_state.vector_db, llm)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a helpful assistant. You will have to answer to user's queries.
        You will have some context to help with your answers, but now always would be completely related or helpful.
        You can also use your knowledge to assist answering the user's queries.\n
        {context}""",
            ),
            MessagesPlaceholder(variable_name="messages"),
            ("user", "{input}"),
        ]
    )
    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)

    print(f"Stuff documents chain: {stuff_documents_chain}")

    return create_retrieval_chain(retriever_chain, stuff_documents_chain)


def stream_llm_rag_response(messages):
    conversation_rag_chain = _get_conversational_rag_chain(llm_stream)
    response_message = "*(RAG Response)*\n"
    for chunk in conversation_rag_chain.pick("answer").stream(
        {"messages": messages[:-1], "input": messages[-1].content}
    ):
        response_message += chunk
        yield chunk

    st.session_state.messages.append({"role": "assistant", "content": response_message})


# Function to stream the response of the LLM
def stream_llm_response(messages):
    response_message = ""

    for chunk in llm_stream.stream(messages):
        response_message += chunk.content
        yield chunk

    st.session_state.messages.append({"role": "assistant", "content": response_message})


def debug_generate_search_query(llm, messages, user_input):
    query_prompt = ChatPromptTemplate.from_messages(
        [
            MessagesPlaceholder(variable_name="messages"),
            ("user", "{input}"),
            (
                "user",
                "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation, focusing on the most recent messages.",
            ),
        ]
    )

    chain = query_prompt | llm | StrOutputParser()

    query = chain.invoke(
        {
            "messages": messages,
            "input": user_input,
        }
    )

    print("🔍 GENERATED SEARCH QUERY:", query)
    return query
