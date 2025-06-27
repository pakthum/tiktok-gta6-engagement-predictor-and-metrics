import streamlit as st
import pandas as pd
import nltk
from google.cloud import storage
import io
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
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
    page_title="TikTok #gta6 Analysis Dashboard",
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

# --- Emotion Detection ---
@st.cache_resource
def load_emotion_model():
    """Load emotion detection model"""
    try:
        # Using a robust emotion detection model
        emotion_classifier = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            return_all_scores=True
        )
        return emotion_classifier
    except Exception as e:
        st.error(f"Failed to load emotion model: {e}")
        return None

def analyze_emotions(texts, emotion_classifier, batch_size=32):
    """Analyze emotions in batch for efficiency"""
    if emotion_classifier is None:
        return [], []
    
    # Filter out empty or invalid texts
    processed_texts = []
    indices = []
    
    for i, text in enumerate(texts):
        if pd.notna(text) and isinstance(text, str) and text.strip():
            # Truncate very long texts to avoid memory issues
            processed_text = str(text)[:512]
            processed_texts.append(processed_text)
            indices.append(i)
    
    if not processed_texts:
        return ['Unknown'] * len(texts), [0.0] * len(texts)
    
    # Process in batches
    all_emotions = []
    all_confidences = []
    
    try:
        for i in range(0, len(processed_texts), batch_size):
            batch = processed_texts[i:i+batch_size]
            batch_results = emotion_classifier(batch)
            
            for result in batch_results:
                # Find the emotion with highest score
                best_emotion = max(result, key=lambda x: x['score'])
                all_emotions.append(best_emotion['label'])
                all_confidences.append(best_emotion['score'])
        
        # Create final results with proper indexing
        final_emotions = ['Unknown'] * len(texts)
        final_confidences = [0.0] * len(texts)
        
        for i, idx in enumerate(indices):
            if i < len(all_emotions):
                final_emotions[idx] = all_emotions[i]
                final_confidences[idx] = all_confidences[i]
        
        return final_emotions, final_confidences
        
    except Exception as e:
        st.error(f"Emotion analysis failed: {e}")
        return ['Unknown'] * len(texts), [0.0] * len(texts)

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

def analyze_with_emotions(df, text_column='caption'):
    """Add emotion analysis to dataframe"""
    emotion_classifier = load_emotion_model()
    
    if emotion_classifier is None:
        df[f'{text_column}_emotion'] = 'Unknown'
        df[f'{text_column}_emotion_confidence'] = 0.0
        return df
    
    with st.spinner(f"Analyzing emotions in {text_column}..."):
        texts = df[text_column].fillna('').astype(str).tolist()
        emotions, confidences = analyze_emotions(texts, emotion_classifier)
        
        df[f'{text_column}_emotion'] = emotions
        df[f'{text_column}_emotion_confidence'] = confidences
    
    return df

# --- Streamlit UI ---
st.title("🎯 TikTok Sentiment & Emotion Analysis Dashboard")
st.markdown("*Comprehensive sentiment analysis, emotion detection, and topic modeling of TikTok content*")

# Sidebar controls
st.sidebar.header("📊 Dashboard Controls")
auto_refresh = st.sidebar.checkbox("Auto-refresh data", value=False)

if auto_refresh:
    st.sidebar.info("Data refreshes every 5 minutes")

# Analysis options
st.sidebar.subheader("🔍 Analysis Options")
analyze_captions = st.sidebar.checkbox("Analyze Caption Emotions", value=True)
analyze_comments = st.sidebar.checkbox("Analyze Comment Emotions", value=True)

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

# Process sentiment data
with st.spinner("Analyzing sentiment..."):
    df = analyze_with_roberta(df)

# Process emotion data
if analyze_captions and 'caption' in df.columns:
    df = analyze_with_emotions(df, 'caption')

if analyze_comments and 'comments' in df.columns:
    # Check if we have comment text column
    comment_text_col = None
    for col in df.columns:
        if 'comment' in col.lower() and 'text' in col.lower():
            comment_text_col = col
            break
    
    if comment_text_col:
        df = analyze_with_emotions(df, comment_text_col)
    else:
        st.sidebar.warning("No comment text column found for emotion analysis")

# Ensure we have the required columns
required_columns = ['comments', 'caption']
missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    st.error(f"❌ Missing required columns: {missing_columns}")
    st.stop()

# Clean the data
df['comments'] = pd.to_numeric(df['comments'], errors='coerce').fillna(0)

# Main metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "📹 Total Videos", 
        f"{len(df):,}"
    )

with col2:
    avg_confidence = df['roberta_confidence'].mean()
    st.metric(
        "🎯 Avg Sentiment Confidence", 
        f"{avg_confidence:.1%}"
    )

with col3:
    if 'caption_emotion_confidence' in df.columns:
        avg_emotion_conf = df['caption_emotion_confidence'].mean()
        st.metric(
            "😊 Avg Emotion Confidence",
            f"{avg_emotion_conf:.1%}"
        )
    else:
        st.metric("😊 Emotion Analysis", "Disabled")

with col4:
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

# Emotion filter
if 'caption_emotion' in df.columns:
    emotion_filter = st.sidebar.multiselect(
        "Filter by Emotion",
        options=df['caption_emotion'].unique(),
        default=df['caption_emotion'].unique()
    )
else:
    emotion_filter = []

# Apply filters
filtered_df = df[df['roberta_sentiment_label'].isin(sentiment_filter)]
if emotion_filter and 'caption_emotion' in df.columns:
    filtered_df = filtered_df[filtered_df['caption_emotion'].isin(emotion_filter)]

# Create tabs for different analyses
tab1, tab2, tab3, tab4 = st.tabs(["📊 Sentiment Analysis", "😊 Emotion Analysis", "🔍 Topic Modeling", "📄 Detailed Data"])

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
    st.header("😊 Emotion Analysis")
    
    if 'caption_emotion' in filtered_df.columns:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎭 Emotion Distribution")
            emotion_counts = filtered_df['caption_emotion'].value_counts()
            
            # Define colors for emotions
            emotion_colors = {
                'joy': '#FFD700',
                'anger': '#FF4500',
                'fear': '#800080',
                'sadness': '#4169E1',
                'surprise': '#FF69B4',
                'disgust': '#228B22',
                'love': '#FF1493',
                'optimism': '#32CD32',
                'pessimism': '#696969',
                'trust': '#87CEEB',
                'anticipation': '#FFA500',
                'Unknown': '#95a5a6'
            }
            
            fig_emotion_pie = px.pie(
                values=emotion_counts.values,
                names=emotion_counts.index,
                title="Emotion Distribution in Captions",
                color=emotion_counts.index,
                color_discrete_map=emotion_colors
            )
            fig_emotion_pie.update_traces(textposition='inside', textinfo='percent+label')
            fig_emotion_pie.update_layout(showlegend=True, height=400)
            st.plotly_chart(fig_emotion_pie, use_container_width=True)
        
        with col2:
            st.subheader("📈 Comments by Emotion")
            
            engagement_by_emotion = filtered_df.groupby('caption_emotion')[
                'comments'
            ].mean().reset_index()
            
            fig_emotion_bar = px.bar(
                engagement_by_emotion,
                x='caption_emotion',
                y='comments',
                title="Average Comments by Emotion",
                color='caption_emotion',
                color_discrete_map=emotion_colors
            )
            fig_emotion_bar.update_layout(
                xaxis_title="Emotion",
                yaxis_title="Average Comments",
                height=400,
                showlegend=False,
                xaxis_tickangle=-45
            )
            st.plotly_chart(fig_emotion_bar, use_container_width=True)
        
        # Emotion-Sentiment Heatmap
        st.subheader("🔥 Emotion vs Sentiment Heatmap")
        
        emotion_sentiment_crosstab = pd.crosstab(
            filtered_df['caption_emotion'],
            filtered_df['roberta_sentiment_label'],
            normalize='index'
        )
        
        fig_heatmap = px.imshow(
            emotion_sentiment_crosstab.values,
            x=emotion_sentiment_crosstab.columns,
            y=emotion_sentiment_crosstab.index,
            aspect='auto',
            title="Emotion-Sentiment Distribution (Normalized by Emotion)",
            color_continuous_scale='RdYlBu'
        )
        fig_heatmap.update_layout(height=400)
        st.plotly_chart(fig_heatmap, use_container_width=True)
        
        # Top emotions by confidence
        st.subheader("🎯 Most Confident Emotion Predictions")
        
        high_confidence_emotions = filtered_df[
            filtered_df['caption_emotion_confidence'] > 0.7
        ].groupby('caption_emotion').agg({
            'caption_emotion_confidence': 'mean',
            'caption': 'count'
        }).round(3).sort_values('caption_emotion_confidence', ascending=False)
        
        high_confidence_emotions.columns = ['Avg Confidence', 'Count']
        st.dataframe(high_confidence_emotions, use_container_width=True)
        
    else:
        st.info("Emotion analysis was not enabled or failed to load. Please enable it in the sidebar.")

with tab3:
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

    # Topics by Sentiment and Emotion
    if lda_topics is not None and doc_topic_probs is not None:
        st.subheader("🎭 Topics by Sentiment & Emotion")
        
        # Assign dominant topic to each document
        dominant_topics = np.argmax(doc_topic_probs, axis=1)
        
        # Create a DataFrame with topics, sentiments, and emotions
        topic_analysis_df = pd.DataFrame({
            'dominant_topic': [f"Topic {i+1}" for i in dominant_topics],
            'sentiment': filtered_df['roberta_sentiment_label'].iloc[:len(dominant_topics)].values
        })
        
        if 'caption_emotion' in filtered_df.columns:
            topic_analysis_df['emotion'] = filtered_df['caption_emotion'].iloc[:len(dominant_topics)].values
        
        # Sentiment heatmap
        topic_sentiment_crosstab = pd.crosstab(
            topic_analysis_df['dominant_topic'], 
            topic_analysis_df['sentiment'],
            normalize='index'
        )
        
        fig_sentiment_heatmap = px.imshow(
            topic_sentiment_crosstab.values,
            x=topic_sentiment_crosstab.columns,
            y=topic_sentiment_crosstab.index,
            aspect='auto',
            title="Topic-Sentiment Distribution"
        )
        fig_sentiment_heatmap.update_layout(height=300)
        st.plotly_chart(fig_sentiment_heatmap, use_container_width=True)
        
        # Emotion heatmap (if available)
        if 'emotion' in topic_analysis_df.columns:
            topic_emotion_crosstab = pd.crosstab(
                topic_analysis_df['dominant_topic'], 
                topic_analysis_df['emotion'],
                normalize='index'
            )
            
            fig_emotion_heatmap = px.imshow(
                topic_emotion_crosstab.values,
                x=topic_emotion_crosstab.columns,
                y=topic_emotion_crosstab.index,
                aspect='auto',
                title="Topic-Emotion Distribution"
            )
            fig_emotion_heatmap.update_layout(height=300)
            st.plotly_chart(fig_emotion_heatmap, use_container_width=True)

with tab4:
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
    
    # Add emotion columns if available
    if 'caption_emotion' in display_df.columns:
        default_cols.extend(['caption_emotion', 'caption_emotion_confidence'])

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
        sentiment_stats = filtered_df.groupby('roberta_sentiment_label').agg({
            'roberta_confidence': ['mean', 'std'],
            'comments': ['mean', 'std']
        }).round(3)
        
        st.dataframe(sentiment_stats)
        
        # Emotion statistics if available
        if 'caption_emotion' in filtered_df.columns:
            st.write("**Emotion Analysis Summary:**")
            emotion_stats = filtered_df.groupby('caption_emotion').agg({
                'caption_emotion_confidence': ['mean', 'std'],
                'comments': ['mean', 'std']
            }).round(3)
            
            st.dataframe(emotion_stats)
            
            # Cross-tabulation of sentiment and emotion
            st.write("**Sentiment-Emotion Cross-Tabulation:**")
            cross_tab = pd.crosstab(
                filtered_df['roberta_sentiment_label'],
                filtered_df['caption_emotion'],
                margins=True
            )
            st.dataframe(cross_tab)

# Footer with analysis insights
st.markdown("---")
st.subheader("🧠 Analysis Insights")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Key Findings:**")
    
    # Sentiment insights
    most_common_sentiment = filtered_df['roberta_sentiment_label'].mode().iloc[0]
    sentiment_percentage = (filtered_df['roberta_sentiment_label'] == most_common_sentiment).mean() * 100
    
    st.write(f"• Most common sentiment: **{most_common_sentiment}** ({sentiment_percentage:.1f}%)")
    
    # Engagement insights
    high_engagement = filtered_df['comments'].quantile(0.75)
    high_engagement_sentiment = filtered_df[filtered_df['comments'] > high_engagement]['roberta_sentiment_label'].mode()
    
    if not high_engagement_sentiment.empty:
        st.write(f"• High-engagement posts are mostly: **{high_engagement_sentiment.iloc[0]}**")
    
    # Emotion insights if available
    if 'caption_emotion' in filtered_df.columns:
        most_common_emotion = filtered_df['caption_emotion'].mode().iloc[0]
        emotion_percentage = (filtered_df['caption_emotion'] == most_common_emotion).mean() * 100
        st.write(f"• Most common emotion: **{most_common_emotion}** ({emotion_percentage:.1f}%)")

with col2:
    st.markdown("**Recommendations:**")
    st.write("• Monitor negative sentiment spikes for community management")
    st.write("• Leverage high-engagement emotions for content strategy")
    st.write("• Use topic modeling insights for targeted campaigns")
    if 'caption_emotion' in filtered_df.columns:
        st.write("• Create content that resonates with dominant emotions")
    st.write("• Track sentiment trends over time for brand monitoring")

# Export functionality
st.markdown("---")
st.subheader("📁 Export Data")

export_cols = st.multiselect(
    "Select columns to export:",
    options=filtered_df.columns.tolist(),
    default=['caption', 'roberta_sentiment_label', 'roberta_confidence', 'comments'] + 
             (['caption_emotion', 'caption_emotion_confidence'] if 'caption_emotion' in filtered_df.columns else [])
)

if export_cols:
    export_df = filtered_df[export_cols]
    
    # Convert to CSV
    csv_data = export_df.to_csv(index=False)
    
    st.download_button(
        label="📥 Download Filtered Data as CSV",
        data=csv_data,
        file_name=f"tiktok_analysis_filtered_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )