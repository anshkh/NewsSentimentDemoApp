import streamlit as st
import pandas as pd
import nltk
import requests
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from bs4 import BeautifulSoup
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from datetime import datetime, timedelta
from urllib.parse import urlparse
import re
import time
import random
from PIL import Image
from io import BytesIO
from datetime import datetime
from datetime import timezone
import numpy as np

st.set_page_config(page_title="News Sentiment Analyzer", page_icon="📰", layout="wide")
# Download NLTK resources
@st.cache_resource
def download_nltk_resources():
    nltk.download('vader_lexicon')
    nltk.download('punkt')
    nltk.download('stopwords')

download_nltk_resources()

# App title and description
st.title("News Sentiment Analyzer")
st.markdown("This app analyzes news articles and provides sentiment analysis.")

# Initialize session state
if 'articles' not in st.session_state:
    st.session_state.articles = None
if 'search_performed' not in st.session_state:
    st.session_state.search_performed = False

# Function to scrape news from Google News
@st.cache_data(ttl=3600)  # Cache for 1 hour
def scrape_google_news(query):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    # For demonstration, we'll create mock data instead of actually scraping
    # In a real application, you would use requests to get data from Google News
    
    # Mock data for demonstration
    mock_articles = [
        {
            "title": "Peace talks resume between ethnic groups in Manipur",
            "source": "Global News Network",
            "date": (datetime.now() - timedelta(days=2)).strftime("%Y-%m-%d"),
            "url": "https://example.com/manipur-peace-talks",
            "snippet": "Representatives from various ethnic communities in Manipur met for the first round of peace talks since January. Officials express cautious optimism about progress made during the discussions."
        },
        {
            "title": "Infrastructure development project launched in Imphal",
            "source": "Economic Times",
            "date": (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d"),
            "url": "https://example.com/manipur-infrastructure",
            "snippet": "The central government has approved a ₹500 crore infrastructure development project for Manipur's capital Imphal, focusing on road connectivity and public facilities."
        },
        {
            "title": "Clashes reported in border areas of Manipur, security forces deployed",
            "source": "Regional Dispatch",
            "date": (datetime.now() - timedelta(days=3)).strftime("%Y-%m-%d"),
            "url": "https://example.com/manipur-border-tension",
            "snippet": "Renewed tensions have been reported along the border areas in Manipur. Security forces have been deployed to prevent escalation as local authorities call for calm."
        },
        {
            "title": "Cultural festival celebrates diversity in Manipur",
            "source": "Arts & Culture Today",
            "date": (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d"),
            "url": "https://example.com/manipur-cultural-festival",
            "snippet": "The annual cultural festival showcasing Manipur's rich heritage opened yesterday with performances from different ethnic communities, promoting unity through cultural exchange."
        },
        {
            "title": "Economic challenges persist in rural Manipur despite aid",
            "source": "Development Monitor",
            "date": (datetime.now() - timedelta(days=4)).strftime("%Y-%m-%d"),
            "url": "https://example.com/manipur-economic-challenges",
            "snippet": "Rural communities in Manipur continue to face economic hardships despite government aid programs. Local officials cite implementation gaps and ongoing security concerns as major factors."
        },
        {
            "title": "Environmental conservation efforts launched in Manipur's forest regions",
            "source": "Environmental News",
            "date": (datetime.now() - timedelta(days=6)).strftime("%Y-%m-%d"),
            "url": "https://example.com/manipur-conservation",
            "snippet": "A coalition of NGOs and government agencies has launched a new initiative to protect Manipur's biodiversity hotspots, with focus on community-based conservation models."
        },
        {
            "title": "New educational initiative brings technology to rural schools in Manipur",
            "source": "Education Weekly",
            "date": (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d"),
            "url": "https://example.com/manipur-education-tech",
            "snippet": "A new program is bringing computers and internet connectivity to rural schools in Manipur, aiming to bridge the digital divide and provide students with modern learning tools."
        },
        {
            "title": "Tourism sector in Manipur shows signs of recovery post-pandemic",
            "source": "Travel Today",
            "date": (datetime.now() - timedelta(days=8)).strftime("%Y-%m-%d"),
            "url": "https://example.com/manipur-tourism-recovery",
            "snippet": "The tourism industry in Manipur is showing positive growth as visitors return to experience the state's unique cultural heritage and natural beauty. Local businesses report increasing bookings."
        },
        {
            "title": "Healthcare accessibility remains a challenge in remote areas of Manipur",
            "source": "Health Reporter",
            "date": (datetime.now() - timedelta(days=9)).strftime("%Y-%m-%d"),
            "url": "https://example.com/manipur-healthcare-challenges",
            "snippet": "Despite recent healthcare initiatives, residents in remote regions of Manipur continue to face difficulties accessing medical services. Officials acknowledge the need for additional mobile clinics."
        },
        {
            "title": "Traditional weaving art of Manipur gaining international recognition",
            "source": "Craft & Design",
            "date": (datetime.now() - timedelta(days=10)).strftime("%Y-%m-%d"),
            "url": "https://example.com/manipur-weaving-recognition",
            "snippet": "The intricate traditional weaving techniques of Manipur are receiving international acclaim, with local artisans being invited to showcase their work at prestigious exhibitions worldwide."
        }
    ]
    url = 'https://newsapi.org/v2/everything?'
    apiKey = '8e9002adec9749ac9a1d90edb9ddf18c'
    results = requests.get(f'{url}q={query}&apiKey={apiKey}').json()
    
    return results

# Function to analyze sentiment using NLTK VADER
def analyze_sentiment(text):
    sid = SentimentIntensityAnalyzer()
    sentiment_scores = sid.polarity_scores(text)
    
    # Categorize sentiment
    compound_score = sentiment_scores['compound']
    if compound_score >= 0.05:
        sentiment = "Positive"
    elif compound_score <= -0.05:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"
    
    return {
        'compound': compound_score,
        'pos': sentiment_scores['pos'],
        'neu': sentiment_scores['neu'],
        'neg': sentiment_scores['neg'],
        'label': sentiment
    }

# Sidebar for settings
st.sidebar.title(f"News Sentiment Analyzer\nSettings")
#num_articles = st.sidebar.slider("Number of articles to analyze", 5, 30, 10)
custom_query = st.sidebar.text_input("Search query", "Manipur news")
#time_period = st.sidebar.selectbox("Time period", ["Past 24 hours", "Past week", "Past month", "Past year"])

# Function to fetch and analyze articles
def fetch_and_analyze():
    with st.spinner('Fetching and analyzing news articles...'):
        # Get articles
        articles = scrape_google_news(custom_query)['articles']
        
        # Analyze sentiment for each article
        for news in articles:
            news_title = news['title']
            news_date = news['publishedAt']
            news_source = news['source']['name']
            news_image = news['urlToImage']
            news_content = news['content']
            news_url = news['url']
            # Combine title and snippet for better sentiment analysis
            full_text = f"{news_title} {news_content}"
            news['sentiment'] = analyze_sentiment(full_text)
        
        st.session_state.articles = articles
        st.session_state.search_performed = True
    
    return articles

# Button to start analysis
if st.sidebar.button("Analyze News"):
    articles = fetch_and_analyze()
else:
    # For demonstration, initialize with some data if first run
    if not st.session_state.search_performed:
        articles = fetch_and_analyze()
    else:
        articles = st.session_state.articles

# Display results if articles have been fetched
if articles:
    # Create dashboard layout
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.subheader("News Articles")
        
        # Filter options
        sentiment_filter = st.multiselect(
            "Filter by sentiment",
            ["Positive", "Neutral", "Negative"],
            default=["Positive", "Neutral", "Negative"]
        )
        
        # Filter articles
        filtered_articles = [a for a in articles if a['sentiment']['label'] in sentiment_filter]
        
        # Display articles
        for i, article in enumerate(filtered_articles):
            sentiment = article['sentiment']['label']
            compound_score = article['sentiment']['compound']
            
            # Determine color based on sentiment
            if sentiment == "Positive":
                color = "#4CAF50"  # Green
            elif sentiment == "Negative":
                color = "#F44336"  # Red
            else:
                color = "#FFC107"  # Amber
            
            date = datetime.fromisoformat(article['publishedAt'][:-1]).astimezone(timezone.utc)
            date.strftime('%Y-%m-%d %H:%M:%S')

            # Article card
            st.markdown(f"""
            <div style="border:1px solid #ddd; border-radius:5px; padding:10px; margin-bottom:10px;">
                <h3>{article['title']}</h3>
                <p style="color:gray; font-size:0.8em;">{article['source']['name']} | {date}</p>
                <p>{article['description']}</p>
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <a href="{article['url']}" target="_blank">Read more</a>
                    <span style="background-color:{color}; color:white; padding:5px 10px; border-radius:15px;">
                        {sentiment} ({compound_score:.2f})
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.subheader("Sentiment Analysis")
        
        # Count sentiments
        sentiment_counts = {"Positive": 0, "Neutral": 0, "Negative": 0}
        for article in articles:
            sentiment_counts[article['sentiment']['label']] += 1
        
        # Create sentiment summary DataFrame
        df_sentiment = pd.DataFrame({
            'Sentiment': list(sentiment_counts.keys()),
            'Count': list(sentiment_counts.values())
        })
        
        # Calculate percentages
        total = df_sentiment['Count'].sum()
        df_sentiment['Percentage'] = df_sentiment['Count'] / total * 100
        
        # Display summary stats
        st.metric("Total Articles", len(articles))
        
        # Create columns for metrics
        met1, met2, met3 = st.columns(3)
        with met1:
            st.metric("Positive", f"{sentiment_counts['Positive']} ({sentiment_counts['Positive']/len(articles)*100:.1f}%)")
        with met2:
            st.metric("Neutral", f"{sentiment_counts['Neutral']} ({sentiment_counts['Neutral']/len(articles)*100:.1f}%)")
        with met3:
            st.metric("Negative", f"{sentiment_counts['Negative']} ({sentiment_counts['Negative']/len(articles)*100:.1f}%)")
        
        # Create pie chart using Plotly
        fig = px.pie(
            df_sentiment, 
            values='Count', 
            names='Sentiment',
            color='Sentiment',
            color_discrete_map={
                'Positive': '#4CAF50',
                'Neutral': '#FFC107',
                'Negative': '#F44336'
            },
            hole=0.4
        )
        fig.update_layout(margin=dict(t=0, b=0, l=0, r=0))
        st.plotly_chart(fig, use_container_width=True)
        
        # Create bar chart for compound scores
        df_scores = pd.DataFrame([
            {
                'Title': article['title'][:30] + '...',
                'Compound Score': article['sentiment']['compound'],
                'Sentiment': article['sentiment']['label']
            }
            for article in articles
        ])
        
        # Sort by compound score
        df_scores = df_scores.sort_values('Compound Score')
        
        # Create bar chart
        fig_bar = px.bar(
            df_scores,
            y='Title',
            x='Compound Score',
            color='Sentiment',
            color_discrete_map={
                'Positive': '#4CAF50',
                'Neutral': '#FFC107',
                'Negative': '#F44336'
            },
            orientation='h',
            title='Article Sentiment Scores'
        )
        fig_bar.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig_bar, use_container_width=True)
        
        # Word cloud for most common words (would go here in a full implementation)
        st.subheader("Data Download")
        
        # Convert articles to DataFrame for download
        df_download = pd.DataFrame([
            {
                'Title': article['title'],
                'Source': article['source'],
                'Date': article['publishedAt'],
                'Snippet': article['description'],
                'URL': article['url'],
                'Sentiment': article['sentiment']['label'],
                'Compound Score': article['sentiment']['compound'],
                'Positive Score': article['sentiment']['pos'],
                'Neutral Score': article['sentiment']['neu'],
                'Negative Score': article['sentiment']['neg']
            }
            for article in articles
        ])
        
        # Add download button
        csv = df_download.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download data as CSV",
            data=csv,
            file_name="news_analysis.csv",
            mime="text/csv",
        )

# Footer
st.markdown("---")
st.markdown("### About This App")
st.markdown("""
This application analyzes news articles and performs sentiment analysis using Natural Language Processing.
It helps track media coverage sentiment trends and provides insights into how people feel about a news.

**Note:** This is a demonstration app.
""")

# Add information about technologies used
st.sidebar.markdown("---")
st.sidebar.markdown("### Technologies Used")
st.sidebar.markdown("""
- Streamlit for the web interface
- NLTK for natural language processing
- VADER for sentiment analysis
- Plotly for interactive visualizations
""")