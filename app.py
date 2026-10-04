import streamlit as st
import os
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import Pinecone as LangChainPinecone
from groq import Groq
from pinecone import Pinecone as NativePineconeClient

st.set_page_config(page_title="Enterprise IT & Security Compliance Portal", layout="centered")
st.title("🛡️ Enterprise Security & IT Compliance Portal")
st.write("Chat with your corporate guidelines in real time. Powered by an automated multi-doc RAG stack.")

# 1. Fetching configurations cleanly
groq_api_key = st.secrets.get("GROQ_API_KEY")
pinecone_api_key = st.secrets.get("PINECONE_API_KEY")
pinecone_index_name = "company-knowledge"

if pinecone_api_key:
    os.environ["PINECONE_API_KEY"] = pinecone_api_key

# 2. Initialize Conversation Chat History Memory
if "messages" not in st.session_state:
    st.session_state.messages = []

# 3. UPGRADE 1: Automated Multi-Format Ingestion (Text & PDFs)
@st.cache_resource
def sync_knowledge_base():
    # Scan current folder directory for policy files
    policy_files = [
        "password_policy.txt", 
        "device_security.txt", 
        "incident_response.txt", 
        "access_control.txt",
        "sample_handbook.pdf" # Place any PDF with this name in your repo to parse it!
    ]
    all_documents = []
    
    for file in policy_files:
        if os.path.exists(file):
            if file.endswith('.pdf'):
                loader = PyPDFLoader(file)
            else:
                loader = TextLoader(file)
            all_documents.extend(loader.load())
            
    if not all_documents:
        return None

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
    chunks = text_splitter.split_documents(all_documents)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
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
    except Exception:
        return None

# Establish background pipeline sync
if groq_api_key and pinecone_api_key:
    with st.spinner("Synchronizing permanent security database..."):
        vector_db = sync_knowledge_base()
else:
    st.error("Configuration keys missing in Streamlit secrets panel!")
    vector_db = None

# 4. UPGRADE 2: ChatGPT-Style Chat Interface
# Render older messages from history log
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Capture new employee questions via chat input box
if user_question := st.chat_input("Ask a compliance question..."):
    # Append user question immediately to the screen layout
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    if not vector_db:
        st.error("Database connection unavailable.")
    else:
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            with st.spinner("Searching internal protocols..."):
                try:
                    # UPGRADE 3: Increased search breadth (k=4) for better coverage
                    retriever = vector_db.as_retriever(search_kwargs={"k": 4})
                    matched_chunks = retriever.invoke(user_question)
                    context_block = "\n\n".join([c.page_content for c in matched_chunks])

                    # Include chat historical thread parameters to give the bot memory
                    chat_history_context = ""
                    for msg in st.session_state.messages[-3:]: # Grab last 3 conversational turns
                        chat_history_context += f"{msg['role'].upper()}: {msg['content']}\n"

                    client = Groq(api_key=groq_api_key)
                    system_instructions = (
                        "You are an expert internal corporate cybersecurity compliance assistant.\n"
                        "Answer the employee query using ONLY the provided text context blocks.\n"
                        "Be direct, highly professional, and cite the specific policy section codes (e.g., SEC-POL-01) where available.\n"
                        "Consider the ongoing conversation context when generating responses.\n\n"
                        f"Recent Conversation Context:\n{chat_history_context}\n"
                        f"Document Context Blocks:\n{context_block}"
                    )

                    response = client.chat.completions.create(
                        model="openai/gpt-oss-20b",
                        messages=[
                            {"role": "system", "content": system_instructions},
                            {"role": "user", "content": user_question}
                        ],
                        temperature=0.0
                    )

                    answer = response.choices[0].message.content
                    message_placeholder.markdown(answer)
                    
                    # Store the bot's response in chat history memory
                    st.session_state.messages.append({"role": "assistant", "content": answer})

                    # UPGRADE 4: Interactive Collapsible Visual Audit Tracer
                    with st.expander("🔍 View Raw Vector Matches (Audit Trace Source Documents)"):
                        for i, chunk in enumerate(matched_chunks):
                            source_name = chunk.metadata.get('source', 'Unknown Document')
                            st.markdown(f"**Match {i+1} From:** `{source_name}`")
                            st.caption(chunk.page_content)
                            st.divider()

                except Exception as err:
                    st.error(f"Internal Pipeline Error: {err}")
