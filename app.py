import streamlit as st
import os
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import Pinecone as LangChainPinecone
from pinecone import Pinecone as NativePineconeClient

st.set_page_config(page_title="Agentic Compliance Officer", layout="centered")
st.title("🤖 Autonomous Agentic Compliance Officer")
st.write("This agent can autonomously choose to search local policy databases or browse the web to answer complex mixed questions.")

# 1. Credentials Setup
groq_api_key = st.secrets.get("GROQ_API_KEY")
pinecone_api_key = st.secrets.get("PINECONE_API_KEY")
pinecone_index_name = "company-knowledge"

if pinecone_api_key:
    os.environ["PINECONE_API_KEY"] = pinecone_api_key

# 2. Database Sync 
@st.cache_resource
def get_vector_db():
    policy_files = ["password_policy.txt", "device_security.txt", "incident_response.txt", "access_control.txt", "sample_handbook.pdf"]
    all_documents = []
    for file in policy_files:
        if os.path.exists(file):
            try:
                loader = TextLoader(file)
                all_documents.extend(loader.load())
            except Exception:
                continue
    if not all_documents: return None
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(all_documents)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    try:
        pc = NativePineconeClient(api_key=pinecone_api_key)
        active_indexes = [idx.name for idx in pc.indexes.list()] if hasattr(pc, 'indexes') else [idx.name for idx in pc.list_indexes()]
        
        if pinecone_index_name not in active_indexes: 
            return None
            
        return LangChainPinecone.from_documents(chunks, embeddings, index_name=pinecone_index_name)
    except Exception:
        return None

vector_db = get_vector_db()

# 3. DEFINE AGENT TOOLS (Autonomous Actions)

@tool
def search_internal_company_policies(query: str) -> str:
    """Use this tool to search internal corporate policy documents regarding passwords, USB device rules, visitor badging, and travel expenses."""
    if not vector_db:
        return "Internal policy database is currently unavailable."
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})
    matched_docs = retriever.invoke(query)
    return "\n\n".join([doc.page_content for doc in matched_docs])

@tool
def search_public_internet_compliance(query: str) -> str:
    """Use this tool ONLY if the internal company policies do not contain the answer, and you need to look up public global cybersecurity standards like SOC2, ISO27001, or NIST guidelines."""
    try:
        search_tool = DuckDuckGoSearchRun()
        return search_tool.invoke(query)
    except Exception as e:
        return f"Web search failed: {e}"

# 4. AGENT LOGIC CORE (The Reasoning Loop)
if "agent_messages" not in st.session_state:
    st.session_state.agent_messages = []

for msg in st.session_state.agent_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_input := st.chat_input("Ask the Agent anything..."):
    st.session_state.agent_messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        status_placeholder = st.empty()
        
        # FIXED LINE BELOW: Swapped out decommissioned model name for an active versatile LLM
        llm = ChatGroq(api_key=groq_api_key, model_name="llama-3.3-70b-versatile", temperature=0.0)
        
        tools = [search_internal_company_policies, search_public_internet_compliance]
        llm_with_tools = llm.bind_tools(tools)
        
        with st.spinner("Agent is reasoning and selecting optimal tools..."):
            try:
                messages = [
                    {"role": "system", "content": "You are an autonomous Agentic Compliance Officer. You must evaluate the user query, choose the appropriate tool(s) to fetch facts, analyze the tool observations, and provide a comprehensive final corporate breakdown."},
                    {"role": "user", "content": user_input}
                ]
                
                response = llm_with_tools.invoke(messages)
                
                if response.tool_calls:
                    for tool_call in response.tool_calls:
                        tool_name = tool_call["name"]
                        tool_args = tool_call["args"]
                        
                        status_placeholder.info(f"🧠 Agent Decision: Executing Tool `{tool_name}` with parameters: {tool_args}")
                        
                        if tool_name == "search_internal_company_policies":
                            q_str = tool_args.get("query", str(tool_args)) if isinstance(tool_args, dict) else str(tool_args)
                            tool_result = search_internal_company_policies.invoke(q_str)
                        elif tool_name == "search_public_internet_compliance":
                            q_str = tool_args.get("query", str(tool_args)) if isinstance(tool_args, dict) else str(tool_args)
                            tool_result = search_public_internet_compliance.invoke(q_str)
                        else:
                            tool_result = "Unknown tool execution attempt."
                            
                        messages.append(response)
                        messages.append({"role": "tool", "tool_call_id": tool_call["id"], "name": tool_name, "content": tool_result})
                    
                    final_response = llm.invoke(messages)
                    answer = final_response.content
                else:
                    answer = response.content
                
                status_placeholder.empty()
                st.markdown(answer)
                st.session_state.agent_messages.append({"role": "assistant", "content": answer})
                
            except Exception as agent_err:
                st.error(f"Agent Execution Loop Crashed: {agent_err}")
