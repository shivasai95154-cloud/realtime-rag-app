import streamlit as st
import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import Pinecone as LangChainPinecone
from groq import Groq
from pinecone import Pinecone as NativePineconeClient

st.set_page_config(page_title="IT & Security Compliance Assistant", layout="centered")
st.title("🛡️ Internal Security & IT Compliance Portal")
st.write("Ask any question about corporate passwords, incident reports, device restrictions, or visitor access.")

# 1. Fetching credentials cleanly
groq_api_key = st.secrets.get("GROQ_API_KEY")
pinecone_api_key = st.secrets.get("PINECONE_API_KEY")
pinecone_index_name = "company-knowledge"

# Force set global system environment parameters
if pinecone_api_key:
    os.environ["PINECONE_API_KEY"] = pinecone_api_key

# 2. Hardened synchronization strategy
@st.cache_resource
def sync_knowledge_base():
    policy_files = ["password_policy.txt", "device_security.txt", "incident_response.txt", "access_control.txt"]
    all_documents = []
    
    for file in policy_files:
        if os.path.exists(file):
            loader = TextLoader(file)
            all_documents.extend(loader.load())
            
    if not all_documents:
        return None

    # Text segment breakdown rules
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(all_documents)
    
    # Vector extraction framework
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Push chunks automatically up to your Pinecone cloud database
    try:
        pc_client = NativePineconeClient(api_key=pinecone_api_key)
        active_indexes = [idx.name for idx in pc_client.list_indexes()]
        
        if pinecone_index_name not in active_indexes:
            return None
            
        vector_db = LangChainPinecone.from_documents(
            documents=chunks, 
            embedding=embeddings, 
            index_name=pinecone_index_name
        )
        return vector_db
    except Exception as network_err:
        return None

# Load portal database parameters
if groq_api_key and pinecone_api_key:
    with st.spinner("Connecting to permanent security database..."):
        vector_db = sync_knowledge_base()
else:
    st.error("Missing configuration keys! Check your Streamlit advanced settings secrets panel.")
    vector_db = None

# 3. Employee UI Interaction Layer
user_question = st.text_input("Enter your security compliance question:")

if user_question and vector_db:
    with st.spinner("Searching compliance documents..."):
        try:
            retriever = vector_db.as_retriever(search_kwargs={"k": 2})
            matched_chunks = retriever.invoke(user_question)
            context_block = "\n\n".join([c.page_content for c in matched_chunks])

            client = Groq(api_key=groq_api_key)
            system_instructions = (
                "You are an expert internal corporate cybersecurity compliance assistant.\n"
                "Answer the employee query using ONLY the provided text context blocks.\n"
                "Be direct, highly professional, and cite the specific policy section codes (e.g., SEC-POL-01) where available.\n"
                f"Context Blocks:\n{context_block}"
            )

            response = client.chat.completions.create(
                model="openai/gpt-oss-20b",
                messages=[
                    {"role": "system", "content": system_instructions},
                    {"role": "user", "content": user_question}
                ],
                temperature=0.0
            )

            st.success("🔒 Official Compliance Response:")
            st.write(response.choices.message.content)

        except Exception as err:
            st.error(f"Internal System Error: {err}")
