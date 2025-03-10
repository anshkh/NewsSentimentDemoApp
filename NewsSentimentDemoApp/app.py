import operator
import streamlit as st
import pickle
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from newsapi import NewsApiClient
import requests, json
import pathlib


st.set_page_config('News Sentiment Analysis')

# Download stopwords once, using Streamlit's caching
@st.cache_resource
def load_stopwords():
    nltk.download('stopwords')
    stoplist = set(stopwords.words("english"))
    return stoplist

# Load model and vectorizer once
@st.cache_resource
def load_model_and_vectorizer():
    with open('model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('vectorizer.pkl', 'rb') as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
    return model, vectorizer

# Define sentiment prediction function
def predict_sentiment(text):
    # Preprocess text
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    text = text.split()
    text = [word for word in text if word not in stopwords.words('english')]
    text = ' '.join(text)
    
    analyser = SentimentIntensityAnalyzer()
   
    score = analyser.polarity_scores(text)
    sentiment = max(score, key=score.get)
    sentiment_value = score[sentiment]
    sentiment_text = 'Neutral'
    if sentiment == 'neg':
        sentiment_text = "Negative"
    elif sentiment == 'neu':
        sentiment_text = "Neutral"
    elif sentiment == 'pos':
        sentiment_text = "Positive"
    elif sentiment == 'compound':
        sentiment_text = "Neutral"
    
    return {'sentiment_text':sentiment_text, 'score':sentiment_value}

def getNewsResult(query):
    url = 'https://newsapi.org/v2/everything?'
    apiKey = '8e9002adec9749ac9a1d90edb9ddf18c'
    results = requests.get(f'{url}q={query}&apiKey={apiKey}').json()

    if  (len(results) > 0):
        return results
    else:
        return dict(articles="")

# Function to create a colored card
def create_card1(tweet_text, sentiment):
    color = "green" if sentiment == "Positive" else "red"

    card_html = f"""
    <div style="background-color: {color}; padding: 10px; border-radius: 5px; margin: 10px 0;">
        <h5 style="color: white;">{sentiment} Sentiment</h5>
        <p style="color: white;">{tweet_text}</p>
    </div>
    """
    return card_html

# Function to create a colored card
def create_card(sentiment,title,date,source,image, description,url):
    color = "blue"
    sentiment = sentiment['sentiment_text']
    score = sentiment[1]
    st.write(score)
    if sentiment == "Positive":
        color = "green"
    elif sentiment=="Negative":
        color = "red"
    elif sentiment=="Neutral":
        color = "yellow"
    else:
        color = "blue"
      
    card_html = f"""
        <div class="projcard-container">
        <div class="projcard projcard-blue">
            <div class="projcard-innerbox">
                <img class="projcard-img" src="{image}" />
                <div class="projcard-textbox">
                    <div class="projcard-title" style="background-color: {color};">{title}</div>
                    <div class="projcard-subtitle">{source} Date: {date}</div>
                    <a href="{url}" target="_blank">Read More</a>
                    <div class="projcard-bar"></div>
                    <div class="projcard-description">{description}</div>
                    <div class="projcard-tagbox">
                        <span class="projcard-tag">{sentiment}({score})</span>
                    </div>
                </div>
            </div>
        </div>
        </div>"""
    return card_html

#Load CSS
def load_css(filepath):
    with open(filepath) as f:
        st.html(f"<style>{f.read()}</style>")
css_path = pathlib.Path("style.css")
load_css(css_path)

# Main app logic
def main():
    
    st.title("News Sentiment Analysis")
   # Load stopwords, model, vectorizer, and scraper only once
    #stop_words = load_stopwords()
    model, vectorizer = load_model_and_vectorizer()
    #scraper = initialize_scraper()

    # User input: either text input or Twitter username
    option = st.selectbox("Choose an option", ["Input text", "Get News on a topic"])
    
    if option == "Input text":
        text_input = st.text_area("Enter text to analyze sentiment")
        if st.button("Analyze"):
            sentiment = predict_sentiment(text_input)
            st.write(f"Sentiment: {sentiment['sentiment_text']},({sentiment['score']})")

    elif option == "Get News on a topic":
        keyword = st.text_input("Enter News Keyword")
        if st.button("Get News"):
            news_results = getNewsResult(keyword)
            #st.write(news_results)
            if 'articles' in news_results:  # Check if the 'articles' exists
                for news in news_results['articles']:
                    news_title = news['title']
                    news_date = news['publishedAt']
                    news_source = news['source']['name']
                    news_image = news['urlToImage']
                    news_content = news['content']
                    news_url = news['url']
                    sentiment = predict_sentiment(news['description'])
                    
                    # Create and display the colored card for the tweet
                    card_html = create_card(sentiment,news_title,news_date,news_source,news_image, news_content,news_url)
                    st.markdown(card_html, unsafe_allow_html=True)
            else:
                st.write("No News found")
if __name__ == "__main__":
    main()