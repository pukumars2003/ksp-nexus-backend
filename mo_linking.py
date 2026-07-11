import os
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity

_local_path = os.path.join(os.path.dirname(__file__), "ksp-nexus-backend", "mo_embeddings.pkl")
_fallback_path = os.path.join(os.path.dirname(__file__), "ksp_prototype_data", "mo_embeddings.pkl")
INDEX_FILE = _local_path if os.path.exists(_local_path) else _fallback_path

_embedding_model = None

def get_embedding_model():
    # Lazy load to avoid slowing down fast app startups
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        _embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    return _embedding_model

def build_mo_index(fir_df, sample_size=1000):
    """
    Builds embeddings using a combination of crime details since Description doesn't exist.
    We sample it to avoid massive embedding times for the prototype.
    """
    print(f"Building embedding index for {sample_size} cases...")
    df_sample = fir_df.dropna(subset=['CrimeHead_Name']).sample(min(sample_size, len(fir_df)))
    
    # Create a pseudo-description for embeddings
    df_sample['MO_Text'] = df_sample['CrimeHead_Name'].astype(str) + " at " + \
                           df_sample['Place of Offence'].astype(str) + ". Description: " + \
                           df_sample['Description'].astype(str)
                           
    texts = df_sample['MO_Text'].tolist()
    
    model = get_embedding_model()
    vectors = model.encode(texts)
    
    df_sample = df_sample.copy()
    df_sample['embedding'] = list(vectors)
    
    # Save to disk
    os.makedirs(os.path.dirname(INDEX_FILE), exist_ok=True)
    with open(INDEX_FILE, 'wb') as f:
        pickle.dump(df_sample, f)
        
    print(f"Saved MO index to {INDEX_FILE}")
    return df_sample

def load_mo_index():
    if os.path.exists(INDEX_FILE):
        with open(INDEX_FILE, 'rb') as f:
            return pickle.load(f)
    return None

def find_similar_mo(query_case_text, df_index, top_k=5, min_similarity=0.2):
    """
    Finds similar cases based on Modus Operandi (description) text.
    Lowered min_similarity threshold because all-MiniLM distances vary.
    """
    model = get_embedding_model()
    query_vec = model.encode([query_case_text])[0]
    
    # Extract vectors
    all_vectors = np.vstack(df_index['embedding'].values)
    
    # Calculate cosine similarity
    sims = cosine_similarity([query_vec], all_vectors)[0]

    df_result = df_index.copy()
    df_result['similarity'] = sims
    
    # Filter and sort
    matches = df_result[df_result['similarity'] >= min_similarity].sort_values(
        'similarity', ascending=False
    ).head(top_k)

    return matches[['CrimeHead_Name', 'Place of Offence', 'FIR_YEAR', 'similarity']]
