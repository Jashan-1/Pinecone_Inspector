import streamlit as st
import pandas as pd
from pinecone import Pinecone, PineconeException
import json
from typing import Dict, List, Optional
import os
from dotenv import load_dotenv
from transformers import AutoModel, AutoProcessor
import PyPDF2
import umap
import torch
from transformers import AutoModelForCausalLM

# Streamlit page configuration
st.set_page_config(
    page_title="Pinecone Inspector",
    page_icon="🔍",
    layout="wide"
)

# Initialize session state
if 'pc_client' not in st.session_state:
    st.session_state.pc_client = None
if 'fetched_vectors' not in st.session_state:
    st.session_state.fetched_vectors = None
if 'parsed_data' not in st.session_state:
    st.session_state.parsed_data = None

# Load environment variables
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

# Model cache
MODEL_CACHE = {}

# Your existing helper functions (safe_str, initialize_pinecone, etc.)
def safe_str(value: any) -> str:
    try:
        if isinstance(value, bytes):
            return value.decode('utf-8', errors='replace')
        return str(value)
    except Exception:
        return "Conversion Error"

def initialize_pinecone(api_key: str) -> Optional[Pinecone]:
    print(f"Attempting to connect with API key: {api_key[:5]}...")  # Masked for safety
    try:
        api_key = api_key.strip()
        if not api_key.isascii():
            st.error("The API key contains invalid characters.")
            return None
        pc = Pinecone(api_key=api_key)
        indexes = pc.list_indexes()
        index_names = indexes.names()
        st.success(f"Successfully connected to Pinecone! Found {len(index_names)} indexes.")
        print(f"Connection successful, found indexes: {index_names}")
        return pc
    except Exception as e:
        st.error(f"Connection Error: {safe_str(e)}")
        print(f"Connection failed: {str(e)}")
        return None

def get_index_info(pc: Pinecone, index_name: str) -> Optional[Dict]:
    try:
        info = pc.describe_index(index_name)
        return {
            "name": safe_str(index_name),
            "dimension": getattr(info, 'dimension', 0),
            "metric": safe_str(getattr(info, 'metric', 'N/A')),
            "pods": getattr(info, 'pods', 0),
            "replicas": getattr(info, 'replicas', 0),
            "status": safe_str(getattr(info, 'status', 'N/A')),
            "metadata_config": getattr(info, 'metadata_config', {})
        }
    except Exception as e:
        st.warning(f"Error fetching details for {index_name}: {safe_str(e)}")
        return None

def fetch_vectors(index, dimension: int, limit: int = 50) -> List[Dict]:
    try:
        query_vector = [0.0] * dimension
        results = index.query(vector=query_vector, top_k=limit, include_metadata=True)
        vectors = []
        for match in results.matches:
            vector_info = {"id": safe_str(match.id), "score": round(float(match.score), 4)}
            if hasattr(match, 'metadata') and match.metadata:
                for key, value in match.metadata.items():
                    vector_info[f"metadata_{safe_str(key)}"] = safe_str(value)
            vectors.append(vector_info)
        if not vectors:
            st.info("0 vectors found in the index.")
        return vectors
    except Exception as e:
        st.error(f"Error fetching vectors: {safe_str(e)}")
        return []

def delete_all_vectors(index) -> bool:
    try:
        index.delete(delete_all=True, namespace="")
        return True
    except Exception as e:
        st.error(f"Error deleting vectors: {safe_str(e)}")
        return False

def inspect_indexes():
    indexes = st.session_state.pc_client.list_indexes().names()
    if not indexes:
        st.info("No indexes found in your Pinecone account.")
        return
    selected_index = st.selectbox("Select an index to inspect:", options=indexes)
    index_info = get_index_info(st.session_state.pc_client, selected_index)
    if index_info:
        st.write(f"**Name:** {index_info['name']}, **Dimension:** {index_info['dimension']}")
        if st.button("Fetch Vectors"):
            index = st.session_state.pc_client.Index(selected_index)
            vectors = fetch_vectors(index, index_info['dimension'])
            st.session_state.fetched_vectors = vectors
            if vectors:
                st.dataframe(pd.DataFrame(vectors))
        if st.session_state.fetched_vectors and st.button("Delete All Vectors"):
            index = st.session_state.pc_client.Index(selected_index)
            delete_all_vectors(index)
            st.success("Vectors deleted!")
            st.session_state.fetched_vectors = None  # Clear after deletion

from transformers import AutoModelForCausalLM  # Add this import at the top

def load_model(model_name):
    """Load and cache Hugging Face models, handling custom code and model types."""
    if model_name not in MODEL_CACHE:
        try:
            if "Phi-4-multimodal-instruct" in model_name:
                st.warning(
                    f"Loading {model_name} requires executing custom code from Hugging Face. "
                    "Enabled with trust_remote_code=True. Review the repository at "
                    "https://hf.co/microsoft/Phi-4-multimodal-instruct for safety. "
                    "Note: Only text processing is supported currently."
                )
                processor = AutoProcessor.from_pretrained(model_name, token=HF_TOKEN, trust_remote_code=True)
                model = AutoModelForCausalLM.from_pretrained(model_name, token=HF_TOKEN, trust_remote_code=True)
            else:
                processor = AutoProcessor.from_pretrained(model_name, token=HF_TOKEN)
                model = AutoModel.from_pretrained(model_name, token=HF_TOKEN)
            
            MODEL_CACHE[model_name] = (processor, model)
            st.success(f"Loaded {model_name} successfully!")
        except Exception as e:
            st.error(f"Error loading model {model_name}: {str(e)}")
            return None, None
    return MODEL_CACHE[model_name]

def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    return text

def generate_embeddings(text, model_name, target_dimension):
    """Generate embeddings with dimension matching the index."""
    processor, model = load_model(model_name)
    if not processor or not model:
        return None
    
    # Adjust inputs based on model requirements
    if "Phi-4-multimodal-instruct" in model_name:
        inputs = processor(text=text, return_tensors="pt", truncation=True, padding=True)
    else:
        inputs = processor(text=text, return_tensors="pt", truncation=True, padding=True)
    
    with torch.no_grad():
        outputs = model(**inputs)
        # For CausalLM, use last hidden state mean as embedding
        embeddings = outputs.logits.mean(dim=1).numpy()[0] if "Phi-4" in model_name else outputs.last_hidden_state.mean(dim=1).numpy()[0]
    
    current_dim = len(embeddings)
    if current_dim != target_dimension:
        if current_dim < target_dimension:
            embeddings = np.pad(embeddings, (0, target_dimension - current_dim), 'constant')
            st.warning(f"Padded embeddings from {current_dim} to {target_dimension} dimensions.")
        else:
            embeddings = embeddings[:target_dimension]
            st.warning(f"Truncated embeddings from {current_dim} to {target_dimension} dimensions.")
    return embeddings

def upload_and_ingest():
    st.markdown("### Upload & Ingest Documents")
    indexes = st.session_state.pc_client.list_indexes().names()
    if not indexes:
        st.info("No indexes found.")
        return
    selected_index = st.selectbox("Select an index to ingest into:", options=indexes)
    index_info = get_index_info(st.session_state.pc_client, selected_index)
    if not index_info:
        return
    
    target_dimension = index_info["dimension"]  # Get index dimension
    st.write(f"Selected index dimension: {target_dimension}")

    models = [
        "microsoft/Phi-4-multimodal-instruct",
        "OpenGVLab/OmniParser-V2.0",
        "colbert-ir/colpali",
        "mPLUG/DocOwl",
        "sentence-transformers/all-MiniLM-L6-v2"
    ]
    selected_model = st.selectbox("Choose a model for embedding:", options=models)
    
    uploaded_files = st.file_uploader("Upload PDFs:", type=["pdf"], accept_multiple_files=True)
    if uploaded_files and st.button("Ingest Documents"):
        index = st.session_state.pc_client.Index(selected_index)
        parsed_data = []
        with st.spinner("Processing and ingesting documents..."):
            for file in uploaded_files:
                text = extract_text_from_pdf(file)
                try:
                    embeddings = generate_embeddings(text, selected_model, target_dimension)
                    if embeddings is None:
                        continue
                    vector_id = f"doc_{file.name}_{len(parsed_data)}"
                    index.upsert(vectors=[(vector_id, embeddings.tolist(), {"source": file.name, "text": text[:500]})])
                    parsed_data.append({"id": vector_id, "text": text[:500], "source": file.name})
                except PineconeException as pe:
                    st.error(f"Failed to upsert vector: {safe_str(pe)}")
                    continue
        st.session_state.parsed_data = parsed_data
        st.success(f"Ingested {len(parsed_data)} documents!")

    if "parsed_data" in st.session_state:
        st.dataframe(pd.DataFrame(st.session_state.parsed_data))
        if st.button("Visualize Vectors"):
            index = st.session_state.pc_client.Index(selected_index)
            vectors = [index.fetch([v["id"]])["vectors"][v["id"]].values for v in st.session_state.parsed_data]
            reducer = umap.UMAP(n_components=2)
            embedding_2d = reducer.fit_transform(vectors)
            df_viz = pd.DataFrame(embedding_2d, columns=["x", "y"])
            st.scatter_chart(df_viz)

        query = st.text_input("Enter a search query:")
        if st.button("Search"):
            embeddings = generate_embeddings(query, selected_model, target_dimension)  # Pass target_dimension here
            if embeddings is not None:
                index = st.session_state.pc_client.Index(selected_index)
                results = index.query(vector=embeddings.tolist(), top_k=5, include_metadata=True)
                df_results = pd.DataFrame([{"id": m.id, "score": m.score, "text": m.metadata.get("text", "N/A")} for m in results.matches])
                st.dataframe(df_results)

def main():
    st.title("🔍 Pinecone Inspector")
    
    with st.sidebar:
        st.markdown("### Navigation")
        mode = st.radio("Choose an option:", ("Inspect Indexes", "Upload & Ingest Documents"))
        api_key = st.text_input("Enter your Pinecone API Key:", type="password", key="api_key_input")
        
        # Debug output
        print(f"API key entered: {bool(api_key)}, pc_client exists: {'pc_client' in st.session_state}, pc_client value: {st.session_state.get('pc_client')}")
        
        if api_key:
            if 'pc_client' not in st.session_state or st.session_state.pc_client is None:
                with st.spinner("Connecting to Pinecone..."):
                    pc = initialize_pinecone(api_key)
                    st.session_state.pc_client = pc  # Set even if None for clarity
                    if pc:
                        st.success("Connected to Pinecone!")
                        st.rerun()  # Refresh UI after successful connection
                    else:
                        st.error("Failed to connect. Please check your API key.")
            elif st.session_state.pc_client:
                st.success("Already connected to Pinecone!")
        else:
            if 'pc_client' in st.session_state and st.session_state.pc_client is None:
                del st.session_state.pc_client  # Clear invalid state
            st.info("Please enter a Pinecone API key.")

    # Proceed only if pc_client is valid
    if 'pc_client' in st.session_state and st.session_state.pc_client is not None:
        if mode == "Inspect Indexes":
            inspect_indexes()
        else:
            upload_and_ingest()
    else:
        st.info("Waiting for a valid Pinecone connection...")
if __name__ == "__main__":
    main()