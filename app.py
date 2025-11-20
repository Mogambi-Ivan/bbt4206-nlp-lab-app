import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download("stopwords")
nltk.download("punkt")

# Load models and vectorizers

topic_model = joblib.load("model/topic_model_lda.pkl")
topic_vectorizer = joblib.load("model/topic_vectorizer.pkl")
sentiment_model = joblib.load("model/sentiment_classifier_binary_balanced.pkl")
sentiment_vectorizer = joblib.load("model/topic_vectorizer_binary_balanced.pkl")

stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()

def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    tokens = nltk.word_tokenize(text)
    tokens = [w for w in tokens if w not in stop_words]
    tokens = [stemmer.stem(w) for w in tokens]
    return " ".join(tokens)

TOPIC_LABELS = {
    0: "Topic 1: Group work, engagement & quizzes",
    1: "Topic 2: Teaching, content delivery & notes",
    2: "Topic 3: Assessments, difficulty & workload",
    3: "Topic 4: Practical labs, journals & support",
    4: "Topic 5: Real-world BI application",
}

def predict_topic_and_sentiment(raw_text: str):
    cleaned = clean_text(raw_text)
    X_topic = topic_vectorizer.transform([cleaned])
    topic_distribution = topic_model.transform(X_topic)[0]
    topic_idx = topic_distribution.argmax()
    topic_label = TOPIC_LABELS.get(topic_idx, f"Topic {topic_idx + 1}")
    X_sent = sentiment_vectorizer.transform([cleaned])
    sentiment_label = sentiment_model.predict(X_sent)[0]
    return topic_label, sentiment_label

st.title("BI Course Evaluation – Topic & Sentiment Demo")
st.write(
    "Type or paste a student course evaluation comment below. "
    "The app will predict the dominant topic and sentiment."
)

user_input = st.text_area(
    "Student comment",
    placeholder="e.g. The teacher explained the BI concepts clearly and the labs were very practical.",
    height=150,
)

if st.button("Analyse"):
    if not user_input.strip():
        st.warning("Please enter a comment first.")
    else:
        topic, sentiment = predict_topic_and_sentiment(user_input)
        st.subheader("Results")
        st.markdown(f"**Predicted Topic:** {topic}")
        st.markdown(f"**Predicted Sentiment:** {sentiment}")
