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
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans
import re
from collections import Counter
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

# --- Google Cloud Cred CONFIG ---
BUCKET_NAME = "tiktok-sentiment-data"
CSV_FILENAME = "tiktok_gta6_data.csv"

# --- Text Preprocessing ---
@st.cache_data
def preprocess_text(text_series):
    """Clean and preprocess text for topic modeling"""
    def clean_text(text):
        if pd.isna(text) or not isinstance(text, str):
            return ""
        text = text.lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'@\w+|#\w+', '', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = ' '.join(text.split())
        
        return text
    return text_series.apply(clean_text)

# --- TF-IDF Analysis ---
@st.cache_data
def perform_tfidf_analysis(texts, max_features=100, ngram_range=(1, 2)):
    """Perform TF-IDF analysis to find important terms"""
    clean_texts = [text for text in texts if text.strip()]
    
    if not clean_texts:
        return None, None, None
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
        mean_scores = np.array(tfidf_matrix.mean(axis=0)).flatten()
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
    clean_texts = [text for text in texts if text.strip()]
    
    if not clean_texts or len(clean_texts) < n_topics:
        return None, None, None

    stop_words = set(stopwords.words('english'))
    custom_stops = {'tiktok', 'video', 'gta', 'gta6', 'game', 'gaming', 'like', 'get', 'would', 'could', 'really', 'one', 'go', 'see', 'know', 'think', 'good', 'bad', 'way', 'time', 'make', 'come', 'want', 'need'}
    stop_words.update(custom_stops)
    
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
@st.cache_data(ttl=300)  
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
        emotion_classifier = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            return_all_scores=True
        )
        return emotion_classifier
    except Exception as e:
        st.error(f"Failed to load emotion model: {e}")
        return None

def split_comments(comment_text, separator="|||"):
    """Split comment text by separator and clean individual comments"""
    if pd.isna(comment_text) or not isinstance(comment_text, str):
        return []
    
    comments = [comment.strip() for comment in comment_text.split(separator)]
    comments = [comment for comment in comments if comment]
    return comments

def analyze_emotions_individual(texts, emotion_classifier, batch_size=32):
    """Analyze emotions for individual comments, handling |||separated comments"""
    if emotion_classifier is None:
        return []
    
    all_individual_comments = []
    comment_to_original_mapping = [] 
    for original_idx, text in enumerate(texts):
        individual_comments = split_comments(text)
        for comment in individual_comments:
            if comment.strip():  

                processed_comment = comment[:512]
                all_individual_comments.append(processed_comment)
                comment_to_original_mapping.append(original_idx)
    
    if not all_individual_comments:
        return [{'comments': [], 'emotions': [], 'confidences': []}] * len(texts)
    

    try:
        all_emotions = []
        all_confidences = []
        
        for i in range(0, len(all_individual_comments), batch_size):
            batch = all_individual_comments[i:i+batch_size]
            batch_results = emotion_classifier(batch)
            
            for result in batch_results:
                best_emotion = max(result, key=lambda x: x['score'])
                all_emotions.append(best_emotion['label'])
                all_confidences.append(best_emotion['score'])
        

        results = []
        for original_idx in range(len(texts)):
            original_comments = []
            original_emotions = []
            original_confidences = []
            
            for comment_idx, mapped_original_idx in enumerate(comment_to_original_mapping):
                if mapped_original_idx == original_idx:
                    if comment_idx < len(all_individual_comments):
                        original_comments.append(all_individual_comments[comment_idx])
                    if comment_idx < len(all_emotions):
                        original_emotions.append(all_emotions[comment_idx])
                        original_confidences.append(all_confidences[comment_idx])
            
            results.append({
                'comments': original_comments,
                'emotions': original_emotions,
                'confidences': original_confidences,
                'total_comments': len(original_comments)
            })
        
        return results
        
    except Exception as e:
        st.error(f"Individual emotion analysis failed: {e}")
        return [{'comments': [], 'emotions': [], 'confidences': [], 'total_comments': 0}] * len(texts)

def get_emotion_summary(emotion_results):
    """Get summary statistics for emotions from individual comment analysis"""
    summaries = []
    
    for result in emotion_results:
        emotions = result.get('emotions', [])
        confidences = result.get('confidences', [])
        
        if not emotions:
            summaries.append({
                'dominant_emotion': 'Unknown',
                'avg_confidence': 0.0,
                'emotion_distribution': {},
                'total_comments': 0
            })
            continue
        
        emotion_counts = {}
        for emotion in emotions:
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        dominant_emotion = max(emotion_counts.items(), key=lambda x: x[1])[0]
        
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        total = len(emotions)
        emotion_distribution = {emotion: (count/total)*100 
                             for emotion, count in emotion_counts.items()}
        
        summaries.append({
            'dominant_emotion': dominant_emotion,
            'avg_confidence': avg_confidence,
            'emotion_distribution': emotion_distribution,
            'total_comments': total
        })
    
    return summaries

def process_tiktok_emotions(df, comment_column='comments'):
    """Process TikTok video comments for emotion analysis"""
    
    emotion_classifier = load_emotion_model()
    if emotion_classifier is None:
        return df
    
    comment_texts = df[comment_column].tolist()

    st.info("Analyzing emotions for individual comments...")
    emotion_results = analyze_emotions_individual(comment_texts, emotion_classifier)
    
    emotion_summaries = get_emotion_summary(emotion_results)
    
    df['individual_comment_emotions'] = emotion_results
    df['dominant_emotion'] = [summary['dominant_emotion'] for summary in emotion_summaries]
    df['avg_emotion_confidence'] = [summary['avg_confidence'] for summary in emotion_summaries]
    df['total_analyzed_comments'] = [summary['total_comments'] for summary in emotion_summaries]
    
    all_emotions = set()
    for summary in emotion_summaries:
        all_emotions.update(summary['emotion_distribution'].keys())
    
    for emotion in all_emotions:
        df[f'{emotion}_percentage'] = [
            summary['emotion_distribution'].get(emotion, 0) 
            for summary in emotion_summaries
        ]
    
    return df

def display_individual_emotions(emotion_results, video_index=0):
    """Display individual comment emotions for a specific video"""
    if video_index < len(emotion_results):
        result = emotion_results[video_index]
        st.write(f"Video {video_index + 1} - Individual Comment Analysis:")
        
        for i, (comment, emotion, confidence) in enumerate(
            zip(result['comments'], result['emotions'], result['confidences'])
        ):
            st.write(f"Comment {i+1}: {emotion} ({confidence:.2f})")
            st.write(f"Text: {comment[:100]}...")  # Show first 100 chars
            st.write("---")

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
    emotion_classifier = load_emotion_model()
    
    if emotion_classifier is None:
        df[f'{text_column}_emotion'] = 'Unknown'
        df[f'{text_column}_emotion_confidence'] = 0.0
        return df
    
    with st.spinner(f"Analyzing emotions in {text_column}..."):
        texts = df[text_column].fillna('').astype(str).tolist()
        emotion_results = analyze_emotions_individual(texts, emotion_classifier)
        emotion_summaries = get_emotion_summary(emotion_results)
        
        df[f'{text_column}_emotion'] = [summary['dominant_emotion'] for summary in emotion_summaries]
        df[f'{text_column}_emotion_confidence'] = [summary['avg_confidence'] for summary in emotion_summaries]
    
    return df

def show_explanation_box(title, content):
    """Display a styled explanation box"""
    st.markdown(f"""
    <div style="background-color: #f0f2f6; padding: 15px; border-radius: 10px; margin: 10px 0; border-left: 4px solid #4CAF50;">
        <h4 style="color: #333; margin-top: 0;">💡 {title}</h4>
        <p style="color: #555; margin-bottom: 0;">{content}</p>
    </div>
    """, unsafe_allow_html=True)

# --- Streamlit UI ---
st.title("TikTok #gta6 Analysis Dashboard")
st.markdown("*This dashboard displays metrics related to the #gta6 hashtag page on TikTok. All data is scraped in accordance to TikTok privacy conditions.*")

with st.expander("ℹ️ How to Read This Dashboard", expanded=False):
    st.markdown("""
    **Dashboard Overview:**
    - **Sentiment Analysis**: Determines if content is Positive, Negative, or Neutral
    - **Emotion Analysis**: Identifies specific emotions like joy, anger, fear, etc.
    - **Topic Modeling**: Discovers what themes people are discussing about GTA6
    - **Engagement**: Measures how much interaction (comments, likes) content receives
    
    **Key Metrics Explained:**
    - **Sentiment Confidence**: How certain the AI is about the sentiment (higher = more reliable)
    - **Emotion Confidence**: How certain the AI is about the emotion detected
    - **Average Comments**: Mean number of comments per video in each category
    """)

st.sidebar.header("Dashboard Controls")
auto_refresh = st.sidebar.checkbox("Auto-refresh data", value=False)

if auto_refresh:
    st.sidebar.info("Data refreshes every 5 minutes")

st.sidebar.subheader("Analysis Options")
analyze_captions = st.sidebar.checkbox("Analyze Caption Emotions", value=True)
analyze_comments = st.sidebar.checkbox("Analyze Comment Emotions", value=True)

st.sidebar.subheader("Topic Analysis Settings")
analysis_text = st.sidebar.selectbox(
    "Analyze text from:",
    ["captions", "comments", "both"]
)

n_topics = st.sidebar.slider("Number of LDA Topics", 3, 10, 5)
max_features = st.sidebar.slider("Max Features", 50, 500, 100)

with st.spinner("Loading data from Google Cloud Storage..."):
    df, error = load_csv_from_gcs(BUCKET_NAME, CSV_FILENAME)

if error:
    st.error(f"❌ Error loading data: {error}")
    st.stop()

if df is None or df.empty:
    st.warning("⚠️ No data found")
    st.stop()

with st.spinner("Analyzing sentiment..."):
    df = analyze_with_roberta(df)

if analyze_captions and 'caption' in df.columns:
    df = analyze_with_emotions(df, 'caption')

if analyze_comments and 'comments' in df.columns:
    comment_text_col = None
    for col in df.columns:
        if 'comment' in col.lower() and 'text' in col.lower():
            comment_text_col = col
            break
    
    if comment_text_col:
        df = analyze_with_emotions(df, comment_text_col)
    else:
        st.sidebar.warning("No comment text column found for emotion analysis")

required_columns = ['comments', 'caption']
missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    st.error(f"❌ Missing required columns: {missing_columns}")
    st.stop()

df['comments'] = pd.to_numeric(df['comments'], errors='coerce').fillna(0)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "📹 Total Videos", 
        f"{len(df):,}"
    )

with col2:
    avg_confidence = df['roberta_confidence'].mean()
    st.metric(
        "Avg. Sentiment Confidence", 
        f"{avg_confidence:.1%}",
        help="How confident our AI is in sentiment predictions. Higher = more reliable results."
    )

with col3:
    if 'caption_emotion_confidence' in df.columns:
        avg_emotion_conf = df['caption_emotion_confidence'].mean()
        st.metric(
            "Avg Emotion Confidence",
            f"{avg_emotion_conf:.1%}",
            help="How confident our AI is in emotion predictions. Higher = more reliable results."
        )
    else:
        st.metric("Emotion Analysis", "Disabled")

with col4:
    avg_comments = df['comments'].mean()
    st.metric(
        "Avg Comments", 
        f"{avg_comments:,.0f}",
        help="Average number of comments per video. Higher = more engaging content."
    )

st.sidebar.subheader("🔍 Filters")
sentiment_filter = st.sidebar.multiselect(
    "Filter by Sentiment",
    options=df['roberta_sentiment_label'].unique(),
    default=df['roberta_sentiment_label'].unique()
)

if 'caption_emotion' in df.columns:
    emotion_filter = st.sidebar.multiselect(
        "Filter by Emotion",
        options=df['caption_emotion'].unique(),
        default=df['caption_emotion'].unique()
    )
else:
    emotion_filter = []

filtered_df = df[df['roberta_sentiment_label'].isin(sentiment_filter)]
if emotion_filter and 'caption_emotion' in df.columns:
    filtered_df = filtered_df[filtered_df['caption_emotion'].isin(emotion_filter)]


tab1, tab2, tab3, tab4 = st.tabs(["Sentiment Analysis", "Emotion Analysis", "Topic Modeling", "Detailed Data"])

with tab1:
    st.header("Sentiment Analysis")
    
    show_explanation_box(
        "What is Sentiment Analysis?",
        "Sentiment analysis determines whether text expresses positive, negative, or neutral feelings. We use AI to automatically classify each TikTok caption's overall emotional tone."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Sentiment Distribution")
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
            title="How People Feel About GTA6"
        )
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        fig_pie.update_layout(showlegend=True, height=400)
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        st.subheader("Engagement by Sentiment")
        
        show_explanation_box(
            "Understanding Engagement",
            "This shows how many comments different sentiment types receive on average. Higher bars mean that type of content gets more interaction from viewers."
        )
        
        engagement_by_sentiment = filtered_df.groupby('roberta_sentiment_label')[
            'comments'
        ].mean().reset_index()
        
        fig_bar = px.bar(
            engagement_by_sentiment,
            x='roberta_sentiment_label',
            y='comments',
            title="Average Comments by Sentiment Type",
            color='roberta_sentiment_label',
            color_discrete_map={
                'Positive': '#2ecc71',
                'Neutral': '#f39c12', 
                'Negative': '#e74c3c',
                'Unknown': '#95a5a6'
            }
        )
        fig_bar.update_layout(
            xaxis_title="Sentiment Type",
            yaxis_title="Average Comments",
            height=400,
            showlegend=False
        )
        st.plotly_chart(fig_bar, use_container_width=True)

with tab2:
    st.header("Emotion Analysis")
    
    show_explanation_box(
        "What is Emotion Analysis?",
        "While sentiment tells us if something is positive/negative, emotion analysis identifies specific feelings like joy, anger, fear, or surprise. This gives us deeper insights into how people really feel about GTA6."
    )
    
    if 'caption_emotion' in filtered_df.columns:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Emotion Distribution")
            emotion_counts = filtered_df['caption_emotion'].value_counts()
            
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
                title="Specific Emotions About GTA6",
                color=emotion_counts.index,
                color_discrete_map=emotion_colors
            )
            fig_emotion_pie.update_traces(textposition='inside', textinfo='percent+label')
            fig_emotion_pie.update_layout(showlegend=True, height=400)
            st.plotly_chart(fig_emotion_pie, use_container_width=True)
        
        with col2:
            st.subheader("Comments by Emotion")
            
            engagement_by_emotion = filtered_df.groupby('caption_emotion')[
                'comments'
            ].mean().reset_index()
            
            fig_emotion_bar = px.bar(
                engagement_by_emotion,
                x='caption_emotion',
                y='comments',
                title="Which Emotions Drive More Engagement?",
                color='caption_emotion',
                color_discrete_map=emotion_colors
            )
            fig_emotion_bar.update_layout(
                xaxis_title="Emotion Type",
                yaxis_title="Average Comments",
                height=400,
                showlegend=False,
                xaxis_tickangle=-45
            )
            st.plotly_chart(fig_emotion_bar, use_container_width=True)
        
        st.subheader("Emotion vs Sentiment Relationship")
        
        show_explanation_box(
            "Reading the Heatmap",
            "This heatmap shows how emotions relate to sentiment. Darker colors mean stronger relationships. For example, 'joy' should align with 'positive' sentiment, while 'anger' aligns with 'negative'."
        )
        
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
            title="How Emotions Align with Sentiment (% by Emotion)",
            color_continuous_scale='RdYlBu'
        )
        fig_heatmap.update_layout(height=400)
        st.plotly_chart(fig_heatmap, use_container_width=True)
        
        st.subheader("Most Confident Emotion Predictions")
        
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
    st.header("Topic Modeling Analysis")
    
    show_explanation_box(
        "What is Topic Modeling?",
        "Topic modeling automatically discovers what themes and subjects people are discussing. It groups similar words and phrases to reveal the main conversation topics about GTA6 on TikTok."
    )
    
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
    else:  
        caption_text = filtered_df['caption'].fillna('').astype(str)
        comment_text = filtered_df.get('comment_text', pd.Series([''] * len(filtered_df))).fillna('').astype(str)
        text_to_analyze = caption_text + ' ' + comment_text
        st.info("Analyzing both captions and comments for topics")

    with st.spinner("Preprocessing text..."):
        cleaned_texts = preprocess_text(text_to_analyze)
        cleaned_texts_list = cleaned_texts.tolist()

    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("TF-IDF Analysis")
        
        show_explanation_box(
            "What is TF-IDF?",
            "TF-IDF finds the most important and distinctive words in the text. Higher scores mean words that are both frequent AND unique to this dataset - these are the key terms that make GTA6 discussions special."
        )
        
        with st.spinner("Performing TF-IDF analysis..."):
            tfidf_results, tfidf_vectorizer, tfidf_matrix = perform_tfidf_analysis(
                cleaned_texts_list, max_features=max_features
            )
        
        if tfidf_results is not None:
            top_terms = tfidf_results.head(20)
            
            fig_tfidf = px.bar(
                top_terms.head(15),
                x='tfidf_score',
                y='term',
                orientation='h',
                title="Most Important Terms in GTA6 Discussions",
                color='tfidf_score',
                color_continuous_scale='viridis'
            )
            fig_tfidf.update_layout(height=500, yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_tfidf, use_container_width=True)
        
            with st.expander("View All TF-IDF Terms"):
                st.dataframe(tfidf_results, use_container_width=True)
        else:
            st.error("TF-IDF analysis failed. Please check your data.")

    with col2:
        st.subheader("LDA Topic Modeling")
        
        show_explanation_box(
            "What is LDA Topic Modeling?",
            "LDA (Latent Dirichlet Allocation) automatically groups related words into topics. Each topic represents a theme or subject that people discuss. The words in each topic usually relate to the same concept."
        )
        
        with st.spinner("Performing LDA analysis..."):
            lda_topics, lda_model, doc_topic_probs = perform_lda_analysis(
                cleaned_texts_list, n_topics=n_topics, max_features=max_features
            )
        
        if lda_topics is not None:
            for topic in lda_topics:
                with st.expander(f"Topic {topic['topic_id'] + 1}: {topic['top_words_str']}"):
                    st.markdown(f"**Key theme:** Look at the words below to understand what aspect of GTA6 this topic represents")
                    
                    topic_df = pd.DataFrame({
                        'word': topic['words'][:8],
                        'weight': topic['weights'][:8]
                    })
                    
                    fig_topic = px.bar(
                        topic_df,
                        x='weight',
                        y='word',
                        orientation='h',
                        title=f"Most Important Words in Topic {topic['topic_id'] + 1}"
                    )
                    fig_topic.update_layout(height=300, yaxis={'categoryorder':'total ascending'})
                    st.plotly_chart(fig_topic, use_container_width=True)
            
            if doc_topic_probs is not None:
                st.subheader("Topic Popularity")
                
                show_explanation_box(
                    "Understanding Topic Distribution",
                    "This shows how much each topic appears across all the TikTok content. Higher bars mean that topic is discussed more frequently in the dataset."
                )
                
                avg_topic_probs = np.mean(doc_topic_probs, axis=0)
                topic_dist_df = pd.DataFrame({
                    'Topic': [f"Topic {i+1}" for i in range(len(avg_topic_probs))],
                    'Average_Probability': avg_topic_probs
                })
                
                fig_dist = px.bar(
                    topic_dist_df,
                    x='Topic',
                    y='Average_Probability',
                    title="How Often Each Topic Appears in GTA6 Discussions"
                )
                st.plotly_chart(fig_dist, use_container_width=True)
        else:
            st.error("LDA analysis failed. Please check your data and reduce the number of topics.")
    if lda_topics is not None and doc_topic_probs is not None:
        st.subheader("🎭 Topics by Sentiment & Emotion")

        dominant_topics = np.argmax(doc_topic_probs, axis=1)
        

        topic_analysis_df = pd.DataFrame({
            'dominant_topic': [f"Topic {i+1}" for i in dominant_topics],
            'sentiment': filtered_df['roberta_sentiment_label'].iloc[:len(dominant_topics)].values
        })
        
        if 'caption_emotion' in filtered_df.columns:
            topic_analysis_df['emotion'] = filtered_df['caption_emotion'].iloc[:len(dominant_topics)].values
        

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

    st.subheader("Detailed Data")

    search_term = st.text_input("Search captions:", "")
    if search_term:
        mask = filtered_df['caption'].str.contains(search_term, case=False, na=False)
        display_df = filtered_df[mask]
    else:
        display_df = filtered_df


    default_cols = ['caption', 'roberta_sentiment_label', 'roberta_confidence', 'comments']
    

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


    with st.expander("Summary Statistics"):
        st.write("**Sentiment Analysis Summary:**")
        sentiment_stats = filtered_df.groupby('roberta_sentiment_label').agg({
            'roberta_confidence': ['mean', 'std'],
            'comments': ['mean', 'std']
        }).round(3)
        
        st.dataframe(sentiment_stats)
        

        if 'caption_emotion' in filtered_df.columns:
            st.write("**Emotion Analysis Summary:**")
            emotion_stats = filtered_df.groupby('caption_emotion').agg({
                'caption_emotion_confidence': ['mean', 'std'],
                'comments': ['mean', 'std']
            }).round(3)
            
            st.dataframe(emotion_stats)
            
            st.write("**Sentiment-Emotion Cross-Tabulation:**")
            cross_tab = pd.crosstab(
                filtered_df['roberta_sentiment_label'],
                filtered_df['caption_emotion'],
                margins=True
            )
            st.dataframe(cross_tab)

st.markdown("---")
st.subheader("Analysis Insights")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Key Findings:**")
    

    most_common_sentiment = filtered_df['roberta_sentiment_label'].mode().iloc[0]
    sentiment_percentage = (filtered_df['roberta_sentiment_label'] == most_common_sentiment).mean() * 100
    
    st.write(f"• Most common sentiment: **{most_common_sentiment}** ({sentiment_percentage:.1f}%)")
    
    high_engagement = filtered_df['comments'].quantile(0.75)
    high_engagement_sentiment = filtered_df[filtered_df['comments'] > high_engagement]['roberta_sentiment_label'].mode()
    
    if not high_engagement_sentiment.empty:
        st.write(f"• High-engagement posts are mostly: **{high_engagement_sentiment.iloc[0]}**")
    
    if 'caption_emotion' in filtered_df.columns:
        most_common_emotion = filtered_df['caption_emotion'].mode().iloc[0]
        emotion_percentage = (filtered_df['caption_emotion'] == most_common_emotion).mean() * 100
        st.write(f"• Most common emotion: **{most_common_emotion}** ({emotion_percentage:.1f}%)")

st.markdown("---")
st.subheader("View Full CSV Data") 

default_cols = ['caption', 'roberta_sentiment_label', 'roberta_confidence', 'comments']

if 'comment_text' in filtered_df.columns:
    default_cols.append('comment_text')

if 'caption_emotion' in filtered_df.columns:
    default_cols.extend(['caption_emotion', 'caption_emotion_confidence'])

display_cols = st.multiselect(
    "Select columns to display:",
    options=filtered_df.columns.tolist(),
    default=default_cols,
    key="display_columns_multiselect"
)

if display_cols:
    display_df = filtered_df[display_cols]
        
    col1, col2 = st.columns(2)
    with col1:
        show_index = st.checkbox("Show row indices", value=False)
    with col2:
        max_rows = st.number_input("Max rows to display (0 = all)", min_value=0, value=100)
        
    if max_rows > 0:
        display_data = display_df.head(max_rows)
        if len(display_df) > max_rows:
            st.info(f"Showing first {max_rows} rows out of {len(display_df)} total rows")
    else:
        display_data = display_df
        
    st.dataframe(
        display_data,
        use_container_width=True,
        hide_index=not show_index
    )
    
    st.markdown("### Data Summary")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Rows", len(display_df))
    with col2:
        st.metric("Total Columns", len(display_cols))
    with col3:
        st.metric("Displayed Rows", len(display_data))
        
    st.markdown("### Export Options")
    csv_data = display_df.to_csv(index=show_index)
        
    st.download_button(
        label="📥 Download Displayed Data as CSV",
        data=csv_data,
        file_name=f"tiktok_analysis_display_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )