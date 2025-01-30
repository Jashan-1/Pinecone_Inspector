import streamlit as st
import pandas as pd
from pinecone import Pinecone, PineconeException
import json
from typing import Dict, List, Optional

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

def safe_str(value: any) -> str:
    """Safely convert any value to string, handling Unicode issues"""
    try:
        if isinstance(value, bytes):
            return value.decode('utf-8', errors='replace')
        return str(value)
    except Exception:
        return "Conversion Error"

def initialize_pinecone(api_key: str) -> Optional[Pinecone]:
    """Initialize Pinecone client"""
    try:
        api_key = api_key.strip()
        if not api_key.isascii():
            st.error("The API key contains invalid characters.")
            return None

        pc = Pinecone(api_key=api_key)
        indexes = pc.list_indexes()
        index_names = indexes.names()

        st.success(f"Successfully connected to Pinecone! Found {len(index_names)} indexes.")
        return pc

    except PineconeException as pe:
        st.error(f"Pinecone API Error: {safe_str(pe)}")
        return None
    except Exception as e:
        st.error(f"Connection Error: {safe_str(e)}")
        return None

def get_index_info(pc: Pinecone, index_name: str) -> Optional[Dict]:
    """Fetch index details"""
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
    """Fetch vectors with proper validation"""
    try:
        query_vector = [0.0] * dimension
        results = index.query(
            vector=query_vector,
            top_k=limit,
            include_metadata=True
        )

        vectors = []
        for match in results.matches:
            vector_info = {
                "id": safe_str(match.id),
                "score": round(float(match.score), 4)
            }
            if hasattr(match, 'metadata') and match.metadata:
                for key, value in match.metadata.items():
                    vector_info[f"metadata_{safe_str(key)}"] = safe_str(value)
            vectors.append(vector_info)
        return vectors
    except Exception as e:
        st.error(f"Error fetching vectors: {safe_str(e)}")
        return []

def delete_all_vectors(index) -> bool:
    """Delete all vectors from the index"""
    try:
        # Delete all vectors using namespace=""
        index.delete(delete_all=True, namespace="")
        return True
    except Exception as e:
        st.error(f"Error deleting vectors: {safe_str(e)}")
        return False

def find_duplicate_vectors(vectors: List[Dict]) -> List[Dict]:
    """Find duplicate vectors based on normalized metadata."""
    try:
        if not vectors:
            return []

        seen = {}
        duplicates = []

        for vector in vectors:
            metadata_keys = sorted([k for k in vector.keys() if k.startswith('metadata_')])
            normalized_metadata = tuple((k, str(vector[k]).strip().lower()) for k in metadata_keys)
            vector_key = (
                vector['id'].strip().lower(), 
                normalized_metadata
            )
            
            if vector_key in seen:
                duplicates.append(vector)
            else:
                seen[vector_key] = True

        return duplicates
    except Exception as e:
        st.error(f"Error finding duplicates: {safe_str(e)}")
        return []

def main():
    st.title("🔍 Pinecone Inspector")
    
    with st.sidebar:
        api_key = st.text_input("Enter your Pinecone API Key:", type="password")
        if api_key:
            pc = initialize_pinecone(api_key)
            if pc:
                st.session_state.pc_client = pc

        st.divider()
        st.markdown("### Instructions")
        st.markdown("""
        1. Enter your Pinecone API key
        2. Select an index to inspect
        3. View index details
        4. Click 'Fetch Vectors' to see contents
        5. Use 'Delete All Vectors' with caution
        """)

    if st.session_state.pc_client:
        try:
            indexes = st.session_state.pc_client.list_indexes()
            index_names = indexes.names()

            if not index_names:
                st.info("No indexes found in your Pinecone account.")
                return

            selected_index = st.selectbox("Select an index to inspect:", options=index_names)
            
            if selected_index:
                index_info = get_index_info(st.session_state.pc_client, selected_index)
                
                if index_info:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("### Basic Information")
                        st.write(f"**Name:** {index_info['name']}")
                        st.write(f"**Dimension:** {index_info['dimension']}")
                        st.write(f"**Metric:** {index_info['metric']}")

                    with col2:
                        st.markdown("### Configuration")
                        st.write(f"**Pods:** {index_info['pods']}")
                        st.write(f"**Replicas:** {index_info['replicas']}")
                        st.write(f"**Status:** {index_info['status']}")

                    if index_info['metadata_config']:
                        with st.expander("Metadata Configuration"):
                            st.json(index_info['metadata_config'])
                    
                    if st.button("Fetch Vectors"):
                        with st.spinner("Fetching vectors..."):
                            index = st.session_state.pc_client.Index(selected_index)
                            vectors = fetch_vectors(index, index_info['dimension'])
                            st.session_state.fetched_vectors = vectors
                            
                            if vectors:
                                st.success(f"Found {len(vectors)} vectors.")
                            else:
                                st.info("No vectors found in the index.")
                    
                    if st.session_state.fetched_vectors:
                        st.markdown("### Fetched Vectors")
                        df = pd.DataFrame(st.session_state.fetched_vectors)
                        st.dataframe(df, use_container_width=True)
                        
                        with st.expander("Show Raw Vector Data"):
                            st.json(st.session_state.fetched_vectors)
                        
                        if st.button("Show Duplicate Vectors"):
                            with st.spinner("Checking for duplicates..."):
                                duplicates = find_duplicate_vectors(st.session_state.fetched_vectors)
                                
                                if duplicates:
                                    st.warning(f"Found {len(duplicates)} duplicate vectors.")
                                    df_duplicates = pd.DataFrame(duplicates)
                                    st.markdown("### Duplicate Vectors")
                                    st.dataframe(df_duplicates, use_container_width=True)
                                else:
                                    st.success("No duplicate vectors found.")
                    
                    # Add danger zone section for vector deletion
                    st.markdown("---")
                    with st.expander("⚠️ Danger Zone"):
                        st.markdown("### Delete All Vectors")
                        st.warning("""
                            This action will delete **ALL** vectors and their metadata from the index.
                            The index itself will remain intact. This action cannot be undone.
                        """)
                        
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            confirmation = st.text_input(
                                "Type the index name to confirm deletion",
                                key="delete_confirmation"
                            )
                        
                        with col2:
                            if st.button("Delete All Vectors", type="primary", use_container_width=True):
                                if confirmation == selected_index:
                                    with st.spinner("Deleting all vectors..."):
                                        index = st.session_state.pc_client.Index(selected_index)
                                        if delete_all_vectors(index):
                                            st.success("All vectors have been deleted successfully!")
                                            st.session_state.fetched_vectors = None
                                            st.rerun()
                                else:
                                    st.error("Index name confirmation doesn't match. Please try again.")
                                
        except Exception as e:
            st.error(f"Error: {safe_str(e)}")

if __name__ == "__main__":
    main()