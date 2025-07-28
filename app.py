## RAG Q&A Conversation With PDF Including Chat History
import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import os

from dotenv import load_dotenv
load_dotenv()

#loading environment variables
os.environ['HUGGING_FACE_API_RAG_MEM']=os.getenv("HUGGING_FACE_API_RAG_MEM")
os.environ['GROQ_API_KEY_RAG_MEM']=os.getenv("GROQ_API_KEY_RAG_MEM")
os.environ['LANGCHAIN_API_KEY_RAG_MEM']=os.getenv("LANGCHAIN_API_KEY_RAG_MEM")
os.environ['LANGSMITH_TRACING_RAG_MEM']=os.getenv("LANGSMITH_TRACING_RAG_MEM")
os.environ['LANGSMITH_ENDPOINT_RAG_MEM']=os.getenv("LANGSMITH_ENDPOINT_RAG_MEM")
os.environ['LANGSMITH_API_KEY_RAG_MEM']=os.getenv("LANGSMITH_API_KEY_RAG_MEM")
os.environ['LANGSMITH_PROJECT_RAG_MEM']=os.getenv("LANGSMITH_PROJECT_RAG_MEM")

#embedding
embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

#initialize model
llm=ChatGroq(groq_api_key=os.getenv("GROQ_API_KEY_RAG_MEM"),model_name="Gemma2-9b-It")

## set up Streamlit 
st.title("Conversational RAG With chat history on Cricket")

## chat interface
session_id=st.text_input("Session ID",value="default_session")
if 'store' not in st.session_state:
        st.session_state.store={}



#loading vector store
vectorstore=FAISS.load_local("faiss_vector_store", embeddings,allow_dangerous_deserialization=True)


#creaing retriever
retriever=vectorstore.as_retriever()


contextualize_q_system_prompt=(
            "Given a chat history and the latest user question"
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )

contextualize_q_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", contextualize_q_system_prompt),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}"),
                ]
            )

history_aware_retriever=create_history_aware_retriever(llm,retriever,contextualize_q_prompt)


## Answer question

# Answer question
system_prompt = (
                "You are an assistant for question-answering tasks. "
                "Use the following pieces of retrieved context to answer "
                "the question. If you don't know the answer, say that you "
                "don't know. Use three sentences maximum and keep the "
                "answer concise."
                "\n\n"
                "{context}" )

qa_prompt = ChatPromptTemplate.from_messages(
            [
                    ("system", system_prompt),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}"),
            ]
            )

question_answer_chain=create_stuff_documents_chain(llm,qa_prompt)
rag_chain=create_retrieval_chain(history_aware_retriever,question_answer_chain)


def get_session_history(session:str)->BaseChatMessageHistory:
        if session_id not in st.session_state.store:
            st.session_state.store[session_id]=ChatMessageHistory()
        return st.session_state.store[session_id]
        
conversational_rag_chain=RunnableWithMessageHistory(
            rag_chain,get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )

user_input = st.text_input("Your question:")
if user_input:
    session_history=get_session_history(session_id)
    response = conversational_rag_chain.invoke(
        {"input": user_input},
        config={
            "configurable": {"session_id":session_id}
        },  # constructs a key "abc123" in `store`.
    )
    st.write("Assistant:", response['answer'])

