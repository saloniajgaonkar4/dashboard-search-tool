import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("knowledgebase1.csv")
        # Clean the data
        df = df.dropna(subset=['Description'])  # Remove rows with missing descriptions
        df['Description'] = df['Description'].astype(str).str.strip()  # Clean descriptions
        return df
    except FileNotFoundError:
        st.error("knowledgebase1.csv file not found. Please ensure the file exists.")
        return pd.DataFrame()

# Load model
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

df = load_data()

if not df.empty:
    model = load_model()

    # Create embeddings for all dashboard descriptions
    @st.cache_data
    def create_embeddings(data, _model):
        descriptions = data["Description"].tolist()
        return _model.encode(descriptions, show_progress_bar=True)

    embeddings = create_embeddings(df, model)

    # --- Streamlit UI ---
    st.title("🔎 Dashboard Search Tool")
    st.write(f"Searching through {len(df)} dashboards")

    # Search configuration
    col1, col2 = st.columns([3, 1])
    with col1:
        query = st.text_input("Type your search query:", placeholder="e.g., sales performance, customer analytics")
    with col2:
        num_results = st.selectbox("Number of results:", [3, 5, 10], index=1)

    # Minimum similarity threshold
    min_similarity = st.slider("Minimum similarity threshold:", 0.0, 1.0, 0.1, 0.05)

    if query and query.strip():
        # Clean and encode query
        query_clean = query.strip().lower()
        query_embedding = model.encode([query_clean])
        
        # Compute semantic similarity
        semantic_similarities = cosine_similarity(query_embedding, embeddings)[0]
        
        # Add keyword boosting
        def calculate_keyword_boost(description, query):
            desc_lower = description.lower()
            query_words = query.split()
            
            boost = 0.0
            for word in query_words:
                if len(word) > 2:  # Ignore very short words
                    # Exact match gets highest boost
                    if word in desc_lower:
                        boost += 0.3
                    # Partial match gets smaller boost
                    elif any(word in desc_word for desc_word in desc_lower.split()):
                        boost += 0.15
            
            return min(boost, 0.5)  # Cap the boost at 0.5
        
        # Calculate hybrid scores
        keyword_boosts = np.array([
            calculate_keyword_boost(desc, query_clean) 
            for desc in df['Description'].tolist()
        ])
        
        # Combine semantic similarity with keyword boost
        hybrid_scores = semantic_similarities + keyword_boosts
        
        # Filter results by minimum similarity threshold (using hybrid score)
        valid_indices = np.where(hybrid_scores >= min_similarity)[0]
        
        if len(valid_indices) > 0:
            # Sort by hybrid score and get top results
            sorted_indices = valid_indices[np.argsort(hybrid_scores[valid_indices])[::-1]]
            top_indices = sorted_indices[:num_results]
            results = df.iloc[top_indices].copy()
            results['similarity_score'] = semantic_similarities[top_indices]
            results['keyword_boost'] = keyword_boosts[top_indices]
            results['hybrid_score'] = hybrid_scores[top_indices]
            
            st.subheader(f"🎯 Top {len(results)} Search Results")
            
            # Show search strategy info
            st.info("💡 Results ranked by: Semantic similarity + Keyword matching boost")
            
            for idx, (_, row) in enumerate(results.iterrows()):
                # Create expandable sections for better organization
                with st.expander(f"#{idx+1} - {row['Dashboard']} (Hybrid: {row['hybrid_score']:.3f})", expanded=(idx < 3)):
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.markdown(f"**Description:** {row['Description']}")
                        if 'Link' in row and pd.notna(row['Link']):
                            st.markdown(f"🔗 [Open Dashboard]({row['Link']})")
                        
                        # Highlight matching keywords
                        query_words = query_clean.split()
                        highlighted_desc = row['Description']
                        for word in query_words:
                            if len(word) > 2 and word in row['Description'].lower():
                                highlighted_desc = highlighted_desc.replace(
                                    word, f"**{word}**"
                                ).replace(
                                    word.capitalize(), f"**{word.capitalize()}**"
                                )
                        st.markdown(f"**Highlighted:** {highlighted_desc}")
                    
                    with col2:
                        # Visual similarity indicator
                        hybrid_pct = row['hybrid_score'] * 100
                        semantic_pct = row['similarity_score'] * 100
                        keyword_pct = row['keyword_boost'] * 100
                        
                        st.metric("Hybrid Score", f"{hybrid_pct:.1f}%")
                        st.caption(f"Semantic: {semantic_pct:.1f}%")
                        st.caption(f"Keyword: +{keyword_pct:.1f}%")
                        
                        # Color-coded similarity bar
                        if row['hybrid_score'] >= 0.7:
                            st.success("Excellent match")
                        elif row['hybrid_score'] >= 0.5:
                            st.warning("Good match")
                        else:
                            st.info("Moderate match")
                
                st.divider()
                
        else:
            st.warning(f"No results found with similarity score above {min_similarity:.2f}")
            st.info("Try lowering the similarity threshold or using different keywords.")
            
            # Show some sample queries or dashboard topics
            if len(df) > 0:
                st.write("**Available dashboard topics include:**")
                # Show a few sample descriptions to help users understand what's available
                sample_descriptions = df['Description'].head(3).tolist()
                for desc in sample_descriptions:
                    st.write(f"• {desc[:100]}{'...' if len(desc) > 100 else ''}")

    elif query:
        st.info("Please enter a search query to find relevant dashboards.")

    # Add some helpful information
    with st.sidebar:
        st.header("ℹ️ How to Search")
        st.write("""
        **Tips for better results:**
        - Use specific keywords related to your data needs
        - Try different variations of terms
        - Adjust the similarity threshold if needed
        - Use business terms rather than technical jargon
        - **Exact keyword matches get priority boost!**
        
        **Search combines:**
        - 🧠 Semantic understanding (meaning)
        - 🔍 Keyword matching (exact terms)
        
        **Example queries:**
        - "customer retention analysis"
        - "sales performance metrics" 
        - "financial reporting dashboard"
        - "inventory management"
        """)
        
        if len(df) > 0:
            st.write(f"**Database info:**")
            st.write(f"• Total dashboards: {len(df)}")
            st.write(f"• Columns: {', '.join(df.columns.tolist())}")

else:
    st.error("Could not load data. Please check your CSV file.")