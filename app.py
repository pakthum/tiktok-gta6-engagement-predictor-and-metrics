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
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
warnings.filterwarnings('ignore')

# Configure page
st.set_page_config(
    page_title="TikTok Predictive Analytics Dashboard",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize VADER
@st.cache_resource
def load_vader():
    nltk.download('vader_lexicon', quiet=True)
    return SentimentIntensityAnalyzer()

sid = load_vader()

# --- GCS CONFIG ---
BUCKET_NAME = "tiktok-sentiment-data"
CSV_FILENAME = "tiktok_gta6_data.csv"

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
        return None, None, None
    
    try:
        encoded = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
        with torch.no_grad():
            output = model(**encoded)
        scores = torch.nn.functional.softmax(output.logits, dim=1).squeeze().numpy()
        
        # Get all scores for analysis
        neg_score, neu_score, pos_score = scores[0], scores[1], scores[2]
        label = np.argmax(scores)
        confidence = scores[label]
        
        return label, confidence, {'negative': neg_score, 'neutral': neu_score, 'positive': pos_score}
    except Exception:
        return None, None, None

def analyze_with_roberta(df, text_column='caption'):
    tokenizer, model = load_roberta_model()
    
    results = df[text_column].apply(lambda x: roberta_sentiment_score(str(x), tokenizer, model))
    
    df['roberta_sentiment_label'] = results.apply(
        lambda x: ['Negative', 'Neutral', 'Positive'][x[0]] if x[0] is not None else 'Unknown'
    )
    df['roberta_confidence'] = results.apply(lambda x: x[1] if x[1] is not None else 0.0)
    ## df['roberta_scores'] = results.apply(lambda x: x[2] if x[2] is not None else {})
    
    return df

# --- VADER Sentiment for Comments ---
def analyze_comment_sentiment(df):
    def get_vader_score(text):
        if pd.isna(text) or not isinstance(text, str) or text.strip() == "":
            return 0.0
        return sid.polarity_scores(str(text))['compound']
    
    df['comment_sentiment'] = df['comment_text'].apply(get_vader_score)
    return df

# --- Improved Hype Score Calculation ---
def compute_hype_scores(df):
    # Normalize engagement metrics (0-1 scale)
    engagement_cols = ['likes', 'comments', 'saves']
    
    # Handle missing values and ensure non-negative
    for col in engagement_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        df[col] = df[col].abs()  # Ensure non-negative (though should already be)
    
    # Calculate total engagement
    df['total_engagement'] = df[engagement_cols].sum(axis=1)
    
    # Normalize engagement using log scaling for better distribution
    max_engagement = df['total_engagement'].max()
    if max_engagement > 0:
        df['engagement_normalized'] = np.log1p(df['total_engagement']) / np.log1p(max_engagement)
    else:
        df['engagement_normalized'] = 0
    
    # Sentiment score: Positive=1, Neutral=0.5, Negative=0
    sentiment_weights = {'Positive': 1.0, 'Neutral': 0.5, 'Negative': 0.0, 'Unknown': 0.25}
    df['sentiment_weight'] = df['roberta_sentiment_label'].map(sentiment_weights)
    
    # Normalize comment sentiment from [-1,1] to [0,1]
    df['comment_sentiment_norm'] = (df['comment_sentiment'] + 1) / 2
    
    # Hype Score: combines engagement, sentiment, and confidence
    df['hype_score'] = (
        df['engagement_normalized'] * 0.5 + 
        df['sentiment_weight'] * df['roberta_confidence'] * 0.3 +
        df['comment_sentiment_norm'] * 0.2
    )
    
    # Ensure hype_score is between 0 and 1
    df['hype_score'] = df['hype_score'].clip(0, 1)
    
    return df

# --- NEW: Feature Engineering for Predictive Models ---
def create_features(df):
    """Create features for predictive modeling"""
    
    # Text-based features
    df['caption_length'] = df['caption'].str.len()
    df['word_count'] = df['caption'].str.split().str.len()
    df['exclamation_count'] = df['caption'].str.count('!')
    df['question_count'] = df['caption'].str.count(r'\?')
    df['hashtag_count'] = df['caption'].str.count('#')
    df['mention_count'] = df['caption'].str.count('@')
    df['capital_ratio'] = df['caption'].apply(lambda x: sum(1 for c in str(x) if c.isupper()) / max(len(str(x)), 1))
    
    # Time-based features
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df['hour_of_day'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    # Engagement ratios
    df['comment_like_ratio'] = df['comments'] / (df['likes'] + 1)
    df['save_like_ratio'] = df['saves'] / (df['likes'] + 1)
    
    return df

# --- NEW: Predictive Models ---
@st.cache_data
def train_engagement_models(df):
    """Train models to predict engagement metrics"""
    
    # Prepare features
    feature_cols = [
        'caption_length', 'word_count', 'exclamation_count', 'question_count',
        'hashtag_count', 'mention_count', 'capital_ratio', 'hour_of_day',
        'day_of_week', 'is_weekend', 'roberta_confidence', 'sentiment_weight',
        'comment_sentiment'
    ]
    
    # Remove rows with missing values
    model_df = df.dropna(subset=feature_cols + ['likes', 'comments', 'saves'])
    
    if len(model_df) < 50:  # Need minimum data for training
        return None, None, None, {}
    
    X = model_df[feature_cols]
    
    models = {}
    predictions = {}
    metrics = {}
    
    # Train models for each engagement metric
    for target in ['likes', 'comments', 'saves']:
        y = model_df[target]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train multiple models
        model_types = {
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(random_state=42),
            'Linear Regression': LinearRegression()
        }
        
        best_model = None
        best_score = -np.inf
        
        for name, model in model_types.items():
            model.fit(X_train_scaled, y_train)
            score = model.score(X_test_scaled, y_test)
            
            if score > best_score:
                best_score = score
                best_model = (model, scaler)
        
        models[target] = best_model
        
        # Make predictions
        y_pred = best_model[0].predict(X_test_scaled)
        predictions[target] = {
            'actual': y_test.values,
            'predicted': y_pred
        }
        
        # Calculate metrics
        metrics[target] = {
            'r2': r2_score(y_test, y_pred),
            'mae': mean_absolute_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred))
        }
    
    return models, predictions, X.columns.tolist(), metrics

# --- NEW: Trend Analysis ---
def analyze_trends(df):
    """Analyze engagement trends over time"""
    
    if df['timestamp'].isna().all():
        return None
    
    # Group by date
    daily_trends = df.groupby(df['timestamp'].dt.date).agg({
        'likes': ['mean', 'sum'],
        'comments': ['mean', 'sum'],
        'saves': ['mean', 'sum'],
        'hype_score': 'mean',
        'roberta_sentiment_label': lambda x: (x == 'Positive').mean()
    }).reset_index()
    
    # Flatten column names
    daily_trends.columns = ['date', 'avg_likes', 'total_likes', 'avg_comments', 
                           'total_comments', 'avg_saves', 'total_saves', 
                           'avg_hype', 'positive_ratio']
    
    # Calculate growth rates
    for col in ['total_likes', 'total_comments', 'total_saves']:
        daily_trends[f'{col}_growth'] = daily_trends[col].pct_change()
    
    return daily_trends

# --- NEW: Content Recommendations ---
def generate_content_recommendations(df, models):
    """Generate recommendations for optimal content strategy"""
    
    if models is None:
        return []
    
    recommendations = []
    
    # Analyze high-performing content characteristics
    top_performers = df.nlargest(int(len(df) * 0.1), 'total_engagement')
    
    # Caption length analysis
    optimal_length = top_performers['caption_length'].median()
    recommendations.append(f"📝 Optimal caption length: {optimal_length:.0f} characters")
    
    # Hashtag analysis
    optimal_hashtags = top_performers['hashtag_count'].median()
    recommendations.append(f"#️⃣ Optimal hashtag count: {optimal_hashtags:.0f}")
    
    # Timing analysis
    best_hour = top_performers['hour_of_day'].mode().iloc[0] if not top_performers['hour_of_day'].empty else 12
    recommendations.append(f"🕐 Best posting time: {best_hour}:00")
    
    # Sentiment analysis
    sentiment_performance = df.groupby('roberta_sentiment_label')['total_engagement'].mean()
    best_sentiment = sentiment_performance.idxmax()
    recommendations.append(f"😊 Best performing sentiment: {best_sentiment}")
    
    return recommendations

# --- Streamlit UI ---
st.title("🎯 TikTok Predictive Analytics Dashboard")
st.markdown("*Real-time sentiment analysis with predictive engagement modeling*")

# Sidebar controls
st.sidebar.header("📊 Dashboard Controls")
auto_refresh = st.sidebar.checkbox("Auto-refresh data", value=False)
show_predictions = st.sidebar.checkbox("Show Predictive Analysis", value=True)

if auto_refresh:
    st.sidebar.info("Data refreshes every 5 minutes")

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
with st.spinner("Analyzing sentiment and creating features..."):
    df = analyze_with_roberta(df)
    df = analyze_comment_sentiment(df)
    df = compute_hype_scores(df)
    df = create_features(df)

# Train predictive models
models = None
predictions = None
feature_names = []
model_metrics = {}

if show_predictions and len(df) >= 50:
    with st.spinner("Training predictive models..."):
        models, predictions, feature_names, model_metrics = train_engagement_models(df)

# Main metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "📹 Total Videos", 
        f"{len(df):,}",
        delta=None
    )

with col2:
    avg_confidence = df['roberta_confidence'].mean()
    st.metric(
        "🎯 Avg Confidence", 
        f"{avg_confidence:.1%}",
        delta=None
    )

with col3:
    avg_hype = df['hype_score'].mean()
    st.metric(
        "🔥 Avg Hype Score", 
        f"{avg_hype:.2f}",
        delta=None
    )

with col4:
    avg_engagement = df['total_engagement'].mean()
    st.metric(
        "💬 Avg Engagement", 
        f"{avg_engagement:,.0f}",
        delta=None
    )

# NEW: Predictive Analytics Section
if show_predictions and models is not None:
    st.header("🔮 Predictive Analytics")
    
    # Model Performance
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Model Performance")
        
        perf_data = []
        for metric, scores in model_metrics.items():
            perf_data.append({
                'Metric': metric.title(),
                'R² Score': f"{scores['r2']:.3f}",
                'MAE': f"{scores['mae']:.0f}",
                'RMSE': f"{scores['rmse']:.0f}"
            })
        
        st.dataframe(pd.DataFrame(perf_data), use_container_width=True)
    
    with col2:
        st.subheader("📊 Feature Importance")
        
        if models and 'likes' in models:
            model, scaler = models['likes']
            if hasattr(model, 'feature_importances_'):
                importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=True)
                
                fig_importance = px.bar(
                    importance_df.tail(10),
                    x='Importance',
                    y='Feature',
                    orientation='h',
                    title="Top 10 Features for Likes Prediction"
                )
                st.plotly_chart(fig_importance, use_container_width=True)
    
    # Prediction vs Actual Plots
    if predictions:
        st.subheader("🎯 Prediction Accuracy")
        
        cols = st.columns(3)
        for i, (metric, pred_data) in enumerate(predictions.items()):
            with cols[i]:
                fig_pred = px.scatter(
                    x=pred_data['actual'],
                    y=pred_data['predicted'],
                    title=f"{metric.title()} - Predicted vs Actual",
                    labels={'x': 'Actual', 'y': 'Predicted'}
                )
                
                # Add perfect prediction line
                min_val = min(min(pred_data['actual']), min(pred_data['predicted']))
                max_val = max(max(pred_data['actual']), max(pred_data['predicted']))
                fig_pred.add_trace(
                    go.Scatter(
                        x=[min_val, max_val],
                        y=[min_val, max_val],
                        mode='lines',
                        name='Perfect Prediction',
                        line=dict(dash='dash', color='red')
                    )
                )
                
                st.plotly_chart(fig_pred, use_container_width=True)

# NEW: Content Strategy Recommendations
if show_predictions:
    st.header("💡 Content Strategy Recommendations")
    
    recommendations = generate_content_recommendations(df, models)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Optimization Tips")
        for rec in recommendations:
            st.write(f"• {rec}")
    
    with col2:
        st.subheader("📈 Trend Analysis")
        trends = analyze_trends(df)
        
        if trends is not None and not trends.empty:
            # Show latest trends
            latest_trends = trends.tail(7)  # Last 7 days
            
            fig_trends = px.line(
                latest_trends,
                x='date',
                y=['total_likes', 'total_comments', 'total_saves'],
                title="Last 7 Days Engagement Trends"
            )
            st.plotly_chart(fig_trends, use_container_width=True)

# Filters
st.sidebar.subheader("🔍 Filters")
sentiment_filter = st.sidebar.multiselect(
    "Filter by Sentiment",
    options=df['roberta_sentiment_label'].unique(),
    default=df['roberta_sentiment_label'].unique()
)

# Apply filters
filtered_df = df[df['roberta_sentiment_label'].isin(sentiment_filter)]

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
    st.subheader("📈 Engagement by Sentiment")
    
    engagement_by_sentiment = filtered_df.groupby('roberta_sentiment_label')[
        ['likes', 'comments', 'saves']
    ].mean().reset_index()
    
    fig_bar = px.bar(
        engagement_by_sentiment,
        x='roberta_sentiment_label',
        y=['likes', 'comments', 'saves'],
        title="Average Engagement by Sentiment",
        barmode='group',
        color_discrete_map={
            'likes': '#3498db',
            'comments': '#9b59b6',
            'saves': '#e67e22'
        }
    )
    fig_bar.update_layout(
        xaxis_title="Sentiment",
        yaxis_title="Average Count",
        height=400
    )
    st.plotly_chart(fig_bar, use_container_width=True)

# Time series analysis
if 'timestamp' in filtered_df.columns and not filtered_df['timestamp'].isna().all():
    st.subheader("📅 Engagement Over Time")
    
    # Group by date
    daily_data = filtered_df.groupby(filtered_df['timestamp'].dt.date).agg({
        'likes': 'sum',
        'comments': 'sum', 
        'saves': 'sum',
        'hype_score': 'mean'
    }).reset_index()
    
    fig_time = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Daily Engagement', 'Daily Average Hype Score'),
        specs=[[{"secondary_y": False}], [{"secondary_y": False}]]
    )
    
    # Add engagement traces
    fig_time.add_trace(
        go.Scatter(x=daily_data['timestamp'], y=daily_data['likes'], 
                  name='Likes', line=dict(color='#3498db')),
        row=1, col=1
    )
    fig_time.add_trace(
        go.Scatter(x=daily_data['timestamp'], y=daily_data['comments'], 
                  name='Comments', line=dict(color='#9b59b6')),
        row=1, col=1
    )
    fig_time.add_trace(
        go.Scatter(x=daily_data['timestamp'], y=daily_data['saves'], 
                  name='Saves', line=dict(color='#e67e22')),
        row=1, col=1
    )
    
    # Add hype score
    fig_time.add_trace(
        go.Scatter(x=daily_data['timestamp'], y=daily_data['hype_score'], 
                  name='Hype Score', line=dict(color='#e74c3c', width=3)),
        row=2, col=1
    )
    
    fig_time.update_layout(height=600, showlegend=True)
    fig_time.update_xaxes(title_text="Date", row=2, col=1)
    fig_time.update_yaxes(title_text="Count", row=1, col=1)
    fig_time.update_yaxes(title_text="Hype Score", row=2, col=1)
    
    st.plotly_chart(fig_time, use_container_width=True)

# Hype Score Distribution
st.subheader("🔥 Hype Score Analysis")

col1, col2 = st.columns(2)

with col1:
    fig_hist = px.histogram(
        filtered_df, 
        x='hype_score', 
        nbins=30,
        title="Hype Score Distribution",
        color_discrete_sequence=['#e74c3c']
    )
    fig_hist.update_layout(
        xaxis_title="Hype Score",
        yaxis_title="Count",
        height=400
    )
    st.plotly_chart(fig_hist, use_container_width=True)

with col2:
    # Top performing content
    top_content = filtered_df.nlargest(10, 'hype_score')[
        ['caption', 'hype_score', 'roberta_sentiment_label', 'total_engagement']
    ]
    
    fig_top = px.bar(
        top_content,
        x='hype_score',
        y=range(len(top_content)),
        orientation='h',
        title="Top 10 Content by Hype Score",
        color='roberta_sentiment_label',
        color_discrete_map={
            'Positive': '#2ecc71',
            'Neutral': '#f39c12', 
            'Negative': '#e74c3c'
        }
    )
    fig_top.update_layout(
        yaxis=dict(
            tickmode='array',
            tickvals=list(range(len(top_content))),
            ticktext=[f"{caption[:30]}..." if len(caption) > 30 else caption 
                     for caption in top_content['caption']]
        ),
        height=400
    )
    st.plotly_chart(fig_top, use_container_width=True)

# NEW: Engagement Prediction Tool
if show_predictions and models is not None:
    st.header("🔮 Engagement Predictor")
    st.subheader("Predict engagement for hypothetical content")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        pred_caption_length = st.slider("Caption Length", 0, 500, 100)
        pred_hashtag_count = st.slider("Hashtag Count", 0, 20, 5)
        pred_hour = st.slider("Hour of Day", 0, 23, 12)
        pred_sentiment = st.selectbox("Sentiment", ['Positive', 'Neutral', 'Negative'])
    
    with col2:
        pred_word_count = st.slider("Word Count", 0, 100, 20)
        pred_exclamation = st.slider("Exclamation Marks", 0, 10, 1)
        pred_is_weekend = st.checkbox("Weekend Post")
        pred_confidence = st.slider("Sentiment Confidence", 0.0, 1.0, 0.8)
    
    with col3:
        if st.button("🎯 Predict Engagement"):
            # Create prediction input
            sentiment_weights = {'Positive': 1.0, 'Neutral': 0.5, 'Negative': 0.0}
            
            pred_input = pd.DataFrame({
                'caption_length': [pred_caption_length],
                'word_count': [pred_word_count],
                'exclamation_count': [pred_exclamation],
                'question_count': [0],
                'hashtag_count': [pred_hashtag_count],
                'mention_count': [0],
                'capital_ratio': [0.1],
                'hour_of_day': [pred_hour],
                'day_of_week': [5 if pred_is_weekend else 2],
                'is_weekend': [1 if pred_is_weekend else 0],
                'roberta_confidence': [pred_confidence],
                'sentiment_weight': [sentiment_weights[pred_sentiment]],
                'comment_sentiment': [0.1]
            })
            
            predictions_new = {}
            for metric, (model, scaler) in models.items():
                scaled_input = scaler.transform(pred_input)
                pred_value = model.predict(scaled_input)[0]
                predictions_new[metric] = max(0, int(pred_value))
            
            st.success("🎯 Predicted Engagement:")
            col_pred1, col_pred2, col_pred3 = st.columns(3)
            with col_pred1:
                st.metric("👍 Likes", f"{predictions_new.get('likes', 0):,}")
            with col_pred2:
                st.metric("💬 Comments", f"{predictions_new.get('comments', 0):,}")
            with col_pred3:
                st.metric("💾 Saves", f"{predictions_new.get('saves', 0):,}")

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
default_cols = ['timestamp', 'caption', 'roberta_sentiment_label', 'roberta_confidence', 
               'likes', 'comments', 'saves', 'hype_score']

if show_predictions:
    default_cols.extend(['caption_length', 'hashtag_count', 'hour_of_day'])

columns_to_show = st.multiselect(
    "Select columns to display:",
    options=display_df.columns.tolist(),
    default=[col for col in default_cols if col in display_df.columns]
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
        'hype_score': ['mean', 'std'],
        'total_engagement': ['mean', 'std'],
        'comment_sentiment': ['mean', 'std']
    }).round(3)
    
    st.dataframe(summary_stats)
    
    if show_predictions and models is not None:
        st.write("**Model Performance Summary:**")
        st.dataframe(pd.DataFrame(model_metrics).T.round(3))