# Import for LangChain dependencies
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import retrieval_qa
from sentence_transformers import SentenceTransformer
import faiss
import torch
import streamlit as st
import os
import google.generativeai as genai
from dotenv import load_dotenv
load_dotenv()

class RAG:
    def __init__(self):
        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        self.generation_config = {
            "temperature": 1,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 8192,
            "response_mime_type": "text/plain",
        }
        self.llm()  # Ensure llm() is called during initialization

    def data_preparation(self,uploaded_file):
        temp_file_path = f"temp_{uploaded_file.name}"
        with open(temp_file_path, "wb") as temp_file:
                temp_file.write(uploaded_file.getbuffer())


        loader = PyPDFLoader(temp_file_path)
        docs = loader.load()  
        os.remove(temp_file_path)  
        return docs

    def text_splitter(self,docs):
        splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
        )
        return splitter.split_documents(docs)

    def loading_emb_model(self):
        self.emb_model = SentenceTransformer('all-MiniLM-L6-v2')

    def converting_into_embeddings(self,docs):
        texts = [doc.page_content for doc in docs]
        embeddings = self.emb_model.encode(texts, show_progress_bar=True)
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatL2(dimension)
        self.faiss_index.add(embeddings)
        
        # Store mapping between texts and embeddings
        self.doc_texts = texts

    def llm(self):
        self.model = genai.GenerativeModel(
            model_name="gemini-1.5-pro-002",
            generation_config=self.generation_config,
        )

        self.chat_session = self.model.start_chat(history=[])

    def query_answer(self, query):
            # Convert query into embedding
        query_embedding = self.emb_model.encode([query])
        
        # Retrieve the top K nearest neighbors
        top_k = 3
        _, indices = self.faiss_index.search(query_embedding, top_k)
        relevant_docs = [self.doc_texts[i] for i in indices[0]]
        
        # Formulate the context for the Gemini model
        context = "\n".join(relevant_docs)
        full_prompt = f"Context: {context}\n\nQuery: {query}\n\nAnswer:"
        
        # Get response from the LLM
        response = self.chat_session.send_message(full_prompt)
        return response.text








# Web interface
if __name__ == "__main__":
    obj = RAG()  # Initialize RAG object
    st.title("A product developed by WintaX technologies🔗")

    with st.sidebar:
        st.header("WintaX technologies✅")
        file = st.file_uploader("Upload documents here")
        if file:
            docs = obj.data_preparation(file)
            chunks = obj.text_splitter(docs)
            obj.loading_emb_model()
            obj.converting_into_embeddings(chunks)
            st.success("Document processed and embeddings created!")

    # Chat input loop
    user_query = st.chat_input("Ask AI Anything")  # User input via Streamlit
    if user_query:
        st.chat_message("user").markdown(user_query)  # Display user message
        response = obj.query_answer(user_query)  # Query the RAG model
        if response:
            st.chat_message("assistant").markdown(response)  # Display AI response
