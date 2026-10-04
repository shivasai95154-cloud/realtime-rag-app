import streamlit as st
import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from groq import Groq

st.set_page_config(page_title="Real-Time RAG Tester", layout="centered")
st.title("🤖 Real-Time RAG Playground")
st.write("Upload your `company_policy.txt` file and query it live.")

# Securely fetches the key from Streamlit Secrets
groq_api_key = st.secrets.get("GROQ_API_KEY")
if not groq_api_key:
    groq_api_key = os.environ.get("GROQ_API_KEY")
if not groq_api_key:
    groq_api_key = st.sidebar.text_input("Enter Groq API Key:", type="password")

uploaded_file = st.file_uploader("Upload Text Document (.txt)", type=["txt"])
user_question = st.text_input("Ask a question about the document:")

if uploaded_file and user_question:
    if not groq_api_key:
        st.error("Please provide your Groq API Key to run the model!")
    else:
        with st.spinner("Analyzing document and querying Llama-3..."):
            try:
                temp_file = "temp_rag_data.txt"
                with open(temp_file, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                loader = TextLoader(temp_file)
                docs = loader.load()

                text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                chunks = text_splitter.split_documents(docs)

                embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                vector_db = Chroma.from_documents(chunks, embeddings)

                retriever = vector_db.as_retriever(search_kwargs={"k": 2})
                matched_chunks = retriever.get_relevant_documents(user_question)
                context_block = "\n\n".join([c.page_content for c in matched_chunks])

                client = Groq(api_key=groq_api_key)
                system_instructions = (
                    "You are a strict corporate assistant. Answer the user query using ONLY the provided text block.\n"
                    f"Context Block:\n{context_block}"
                )

                response = client.chat.completions.create(
                    model="llama3-8b-8192",
                    messages=[
                        {"role": "system", "content": system_instructions},
                        {"role": "user", "content": user_question}
                    ],
                    temperature=0.0
                )

                st.success("✨ Verified Answer:")
                st.write(response.choices.message.content)
                os.remove(temp_file)

            except Exception as err:
                st.error(f"Execution Error: {err}")
