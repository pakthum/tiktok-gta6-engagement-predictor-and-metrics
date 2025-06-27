import streamlit as st
import pandas as pd
import nltk
from google.cloud import storage
import io
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Topic modeling imports
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans
import re
from collections import Counter
import wordcloud
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Configure page
st.set_page_config(
    page_title="TikTok Sentiment Analysis Dashboard",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- GCS CONFIG ---
BUCKET_NAME = "tiktok-sentiment-data"
CSV_FILENAME = "tiktok_gta6_data.csv"

# --- Text Preprocessing ---
@st.cache_data
def preprocess_text(text_series):
    """Clean and preprocess text for topic modeling"""
    def clean_text(text):
        if pd.isna(text) or not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs, mentions, hashtags
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    return text_series.apply(clean_text)

# --- TF-IDF Analysis ---
@st.cache_data
def perform_tfidf_analysis(texts, max_features=100, ngram_range=(1, 2)):
    """Perform TF-IDF analysis to find important terms"""
    # Filter out empty texts
    clean_texts = [text for text in texts if text.strip()]
    
    if not clean_texts:
        return None, None, None
    
    # Custom stopwords
    stop_words = set(stopwords.words('english'))
    custom_stops = {'tiktok', 'video', 'gta', 'gta6', 'game', 'gaming', 'like', 'get', 'would', 'could', 'really', 'one', 'go', 'see', 'know', 'think', 'good', 'bad', 'way', 'time', 'make', 'come', 'want', 'need'}
    stop_words.update(custom_stops)
    
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        stop_words=list(stop_words),
        min_df=2,
        max_df=0.8
    )
    
    try:
        tfidf_matrix = vectorizer.fit_transform(clean_texts)
        feature_names = vectorizer.get_feature_names_out()
        
        # Get average TF-IDF scores
        mean_scores = np.array(tfidf_matrix.mean(axis=0)).flatten()
        
        # Create results dataframe
        tfidf_df = pd.DataFrame({
            'term': feature_names,
            'tfidf_score': mean_scores
        }).sort_values('tfidf_score', ascending=False)
        
        return tfidf_df, vectorizer, tfidf_matrix
    except Exception as e:
        st.error(f"TF-IDF analysis failed: {e}")
        return None, None, None

# --- LDA Topic Modeling ---
@st.cache_data
def perform_lda_analysis(texts, n_topics=5, max_features=100):
    """Perform LDA topic modeling"""
    # Filter out empty texts
    clean_texts = [text for text in texts if text.strip()]
    
    if not clean_texts or len(clean_texts) < n_topics:
        return None, None, None
    
    # Custom stopwords
    stop_words = set(stopwords.words('english'))
    custom_stops = {'tiktok', 'video', 'gta', 'gta6', 'game', 'gaming', 'like', 'get', 'would', 'could', 'really', 'one', 'go', 'see', 'know', 'think', 'good', 'bad', 'way', 'time', 'make', 'come', 'want', 'need'}
    stop_words.update(custom_stops)
    
    # Use CountVectorizer for LDA
    vectorizer = CountVectorizer(
        max_features=max_features,
        stop_words=list(stop_words),
        min_df=2,
        max_df=0.8,
        ngram_range=(1, 2)
    )
    
    try:
        doc_term_matrix = vectorizer.fit_transform(clean_texts)
        
        # LDA model
        lda = LatentDirichletAllocation(
            n_components=n_topics,
            random_state=42,
            max_iter=20,
            learning_method='online'
        )
        
        lda.fit(doc_term_matrix)
        
        # Get feature names
        feature_names = vectorizer.get_feature_names_out()
        
        # Extract topics
        topics = []
        for topic_idx, topic in enumerate(lda.components_):
            top_words_idx = topic.argsort()[-10:][::-1]
            top_words = [feature_names[i] for i in top_words_idx]
            top_weights = [topic[i] for i in top_words_idx]
            
            topics.append({
                'topic_id': topic_idx,
                'words': top_words,
                'weights': top_weights,
                'top_words_str': ', '.join(top_words[:5])
            })
        
        # Get document-topic probabilities
        doc_topic_probs = lda.transform(doc_term_matrix)
        
        return topics, lda, doc_topic_probs
    except Exception as e:
        st.error(f"LDA analysis failed: {e}")
        return None, None, None

# --- Load CSV from GCS ---
@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_csv_from_gcs(bucket_name, blob_name):
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        csv_data = blob.download_as_text()
        df = pd.read_csv(io.StringIO(csv_data))
        return df, None
    except Exception as e:
        return None, str(e)

# --- RoBERTa Sentiment Analysis ---
@st.cache_resource
def load_roberta_model():
    MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    return tokenizer, model

def roberta_sentiment_score(text, tokenizer, model):
    if pd.isna(text) or not isinstance(text, str) or text.strip() == "":
        return None, None
    
    try:
        encoded = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
        with torch.no_grad():
            output = model(**encoded)
        scores = torch.nn.functional.softmax(output.logits, dim=1).squeeze().numpy()
        
        label = np.argmax(scores)
        confidence = scores[label]
        
        return label, confidence
    except Exception:
        return None, None

def analyze_with_roberta(df, text_column='caption'):
    tokenizer, model = load_roberta_model()
    
    results = df[text_column].apply(lambda x: roberta_sentiment_score(str(x), tokenizer, model))
    
    df['roberta_sentiment_label'] = results.apply(
        lambda x: ['Negative', 'Neutral', 'Positive'][x[0]] if x[0] is not None else 'Unknown'
    )
    df['roberta_confidence'] = results.apply(lambda x: x[1] if x[1] is not None else 0.0)
    
    return df

# --- Streamlit UI ---
st.title("🎯 TikTok Sentiment Analysis Dashboard")
st.markdown("*Sentiment analysis and topic modeling of TikTok captions with engagement metrics*")

# Sidebar controls
st.sidebar.header("📊 Dashboard Controls")
auto_refresh = st.sidebar.checkbox("Auto-refresh data", value=False)

if auto_refresh:
    st.sidebar.info("Data refreshes every 5 minutes")

# Topic modeling controls
st.sidebar.subheader("🔍 Topic Analysis Settings")
analysis_text = st.sidebar.selectbox(
    "Analyze text from:",
    ["captions", "comments", "both"]
)

n_topics = st.sidebar.slider("Number of LDA Topics", 3, 10, 5)
max_features = st.sidebar.slider("Max Features", 50, 500, 100)

# Load and process data
with st.spinner("Loading data from Google Cloud Storage..."):
    df, error = load_csv_from_gcs(BUCKET_NAME, CSV_FILENAME)

if error:
    st.error(f"❌ Error loading data: {error}")
    st.stop()

if df is None or df.empty:
    st.warning("⚠️ No data found")
    st.stop()

# Process data
with st.spinner("Analyzing sentiment..."):
    df = analyze_with_roberta(df)

# Ensure we have the required columns
required_columns = ['comments', 'caption']
missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    st.error(f"❌ Missing required columns: {missing_columns}")
    st.stop()

# Clean the data
df['comments'] = pd.to_numeric(df['comments'], errors='coerce').fillna(0)

# Main metrics
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "📹 Total Videos", 
        f"{len(df):,}"
    )

with col2:
    avg_confidence = df['roberta_confidence'].mean()
    st.metric(
        "🎯 Avg Confidence", 
        f"{avg_confidence:.1%}"
    )

with col3:
    avg_comments = df['comments'].mean()
    st.metric(
        "💬 Avg Comments", 
        f"{avg_comments:,.0f}"
    )

# Filters
st.sidebar.subheader("🔍 Filters")
sentiment_filter = st.sidebar.multiselect(
    "Filter by Sentiment",
    options=df['roberta_sentiment_label'].unique(),
    default=df['roberta_sentiment_label'].unique()
)

# Apply filters
filtered_df = df[df['roberta_sentiment_label'].isin(sentiment_filter)]

# Create tabs for different analyses
tab1, tab2, tab3 = st.tabs(["📊 Sentiment Analysis", "🔍 Topic Modeling", "📄 Detailed Data"])

with tab1:
    # Interactive visualizations
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Sentiment Distribution")
        sentiment_counts = filtered_df['roberta_sentiment_label'].value_counts()
        
        fig_pie = px.pie(
            values=sentiment_counts.values,
            names=sentiment_counts.index,
            color=sentiment_counts.index,
            color_discrete_map={
                'Positive': '#2ecc71',
                'Neutral': '#f39c12', 
                'Negative': '#e74c3c',
                'Unknown': '#95a5a6'
            },
            title="Sentiment Distribution"
        )
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        fig_pie.update_layout(showlegend=True, height=400)
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        st.subheader("📈 Comments by Sentiment")
        
        engagement_by_sentiment = filtered_df.groupby('roberta_sentiment_label')[
            'comments'
        ].mean().reset_index()
        
        fig_bar = px.bar(
            engagement_by_sentiment,
            x='roberta_sentiment_label',
            y='comments',
            title="Average Comments by Sentiment",
            color='roberta_sentiment_label',
            color_discrete_map={
                'Positive': '#2ecc71',
                'Neutral': '#f39c12', 
                'Negative': '#e74c3c',
                'Unknown': '#95a5a6'
            }
        )
        fig_bar.update_layout(
            xaxis_title="Sentiment",
            yaxis_title="Average Comments",
            height=400,
            showlegend=False
        )
        st.plotly_chart(fig_bar, use_container_width=True)

with tab2:
    st.header("🔍 Topic Modeling Analysis")
    
    # Prepare text for analysis
    if analysis_text == "captions":
        text_to_analyze = filtered_df['caption'].fillna('').astype(str)
        st.info("📝 Analyzing captions for topics")
    elif analysis_text == "comments":
        if 'comment_text' in filtered_df.columns:
            text_to_analyze = filtered_df['comment_text'].fillna('').astype(str)
            st.info("💬 Analyzing comments for topics")
        else:
            st.warning("Comment text column not found. Analyzing captions instead.")
            text_to_analyze = filtered_df['caption'].fillna('').astype(str)
    else:  # both
        caption_text = filtered_df['caption'].fillna('').astype(str)
        comment_text = filtered_df.get('comment_text', pd.Series([''] * len(filtered_df))).fillna('').astype(str)
        text_to_analyze = caption_text + ' ' + comment_text
        st.info("📝💬 Analyzing both captions and comments for topics")

    # Preprocess text
    with st.spinner("Preprocessing text..."):
        cleaned_texts = preprocess_text(text_to_analyze)
        cleaned_texts_list = cleaned_texts.tolist()

    # Topic modeling analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 TF-IDF Analysis")
        with st.spinner("Performing TF-IDF analysis..."):
            tfidf_results, tfidf_vectorizer, tfidf_matrix = perform_tfidf_analysis(
                cleaned_texts_list, max_features=max_features
            )
        
        if tfidf_results is not None:
            # Show top terms
            top_terms = tfidf_results.head(20)
            
            fig_tfidf = px.bar(
                top_terms.head(15),
                x='tfidf_score',
                y='term',
                orientation='h',
                title="Top TF-IDF Terms",
                color='tfidf_score',
                color_continuous_scale='viridis'
            )
            fig_tfidf.update_layout(height=500, yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_tfidf, use_container_width=True)
            
            # Show top terms table
            with st.expander("View All TF-IDF Terms"):
                st.dataframe(tfidf_results, use_container_width=True)
        else:
            st.error("TF-IDF analysis failed. Please check your data.")

    with col2:
        st.subheader("🎯 LDA Topic Modeling")
        with st.spinner("Performing LDA analysis..."):
            lda_topics, lda_model, doc_topic_probs = perform_lda_analysis(
                cleaned_texts_list, n_topics=n_topics, max_features=max_features
            )
        
        if lda_topics is not None:
            # Display topics
            for topic in lda_topics:
                with st.expander(f"Topic {topic['topic_id'] + 1}: {topic['top_words_str']}"):
                    # Create a bar chart for topic words
                    topic_df = pd.DataFrame({
                        'word': topic['words'][:8],
                        'weight': topic['weights'][:8]
                    })
                    
                    fig_topic = px.bar(
                        topic_df,
                        x='weight',
                        y='word',
                        orientation='h',
                        title=f"Topic {topic['topic_id'] + 1} Word Weights"
                    )
                    fig_topic.update_layout(height=300, yaxis={'categoryorder':'total ascending'})
                    st.plotly_chart(fig_topic, use_container_width=True)
            
            # Topic distribution across documents
            if doc_topic_probs is not None:
                st.subheader("📈 Topic Distribution")
                
                # Calculate average topic probabilities
                avg_topic_probs = np.mean(doc_topic_probs, axis=0)
                topic_dist_df = pd.DataFrame({
                    'Topic': [f"Topic {i+1}" for i in range(len(avg_topic_probs))],
                    'Average_Probability': avg_topic_probs
                })
                
                fig_dist = px.bar(
                    topic_dist_df,
                    x='Topic',
                    y='Average_Probability',
                    title="Average Topic Distribution Across Documents"
                )
                st.plotly_chart(fig_dist, use_container_width=True)
        else:
            st.error("LDA analysis failed. Please check your data and reduce the number of topics.")

    # Topics by Sentiment
    if lda_topics is not None and doc_topic_probs is not None:
        st.subheader("🎭 Topics by Sentiment")
        
        # Assign dominant topic to each document
        dominant_topics = np.argmax(doc_topic_probs, axis=1)
        
        # Create a DataFrame with topics and sentiments
        topic_sentiment_df = pd.DataFrame({
            'dominant_topic': [f"Topic {i+1}" for i in dominant_topics],
            'sentiment': filtered_df['roberta_sentiment_label'].iloc[:len(dominant_topics)].values
        })
        
        # Create heatmap data
        topic_sentiment_crosstab = pd.crosstab(
            topic_sentiment_df['dominant_topic'], 
            topic_sentiment_df['sentiment'],
            normalize='index'
        )
        
        fig_heatmap = px.imshow(
            topic_sentiment_crosstab.values,
            x=topic_sentiment_crosstab.columns,
            y=topic_sentiment_crosstab.index,
            aspect='auto',
            title="Topic-Sentiment Distribution (Normalized by Topic)"
        )
        fig_heatmap.update_layout(height=400)
        st.plotly_chart(fig_heatmap, use_container_width=True)

with tab3:
    # Detailed data table
    st.subheader("📄 Detailed Data")

    # Add search functionality
    search_term = st.text_input("🔍 Search captions:", "")
    if search_term:
        mask = filtered_df['caption'].str.contains(search_term, case=False, na=False)
        display_df = filtered_df[mask]
    else:
        display_df = filtered_df

    # Select columns to display
    default_cols = ['caption', 'roberta_sentiment_label', 'roberta_confidence', 'comments']

    columns_to_show = st.multiselect(
        "Select columns to display:",
        options=display_df.columns.tolist(),
        default=default_cols
    )

    if columns_to_show:
        st.dataframe(
            display_df[columns_to_show].round(3),
            use_container_width=True,
            height=400
        )

    # Summary statistics
    with st.expander("📊 Summary Statistics"):
        st.write("**Sentiment Analysis Summary:**")
        summary_stats = filtered_df.groupby('roberta_sentiment_label').agg({
            'roberta_confidence': ['mean', 'std'],
            'comments': ['mean', 'std']
        }).round(3)
        
        st.dataframe(summary_stats)