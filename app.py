
import streamlit as st
import pickle
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# Ensure NLTK data is available (usually done during setup, but good for standalone app)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
try:
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    nltk.download('omw-1.4')

# Load the saved model and TF-IDF vectorizer
@st.cache_resource # Cache the model and vectorizer to avoid reloading on each rerun
def load_artifacts():
    model_filename = 'logistic_regression_sentiment_model.pkl'
    tfidf_vectorizer_filename = 'tfidf_vectorizer.pkl'

    with open(model_filename, 'rb') as file:
        model = pickle.load(file)
    with open(tfidf_vectorizer_filename, 'rb') as file:
        vectorizer = pickle.load(file)
    
    return model, vectorizer

best_model, tfidf_vectorizer = load_artifacts()

# Initialize NLTK components for preprocessing
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Define the preprocessing function (replicated from notebook)
def preprocess_review(review_text, tfidf_vectorizer, lemmatizer, stop_words):
    # Convert to lowercase
    review_text = review_text.lower()

    # Remove HTML tags
    review_text = re.sub('<.*?>', '', review_text)

    # Tokenize
    tokens = word_tokenize(review_text)

    # Remove stopwords
    filtered_tokens = [word for word in tokens if word not in stop_words]

    # Lemmatize
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]

    # Join back into a string for TF-IDF
    preprocessed_text = ' '.join(lemmatized_tokens)

    # Transform with the fitted TF-IDF vectorizer
    tfidf_feature = tfidf_vectorizer.transform([preprocessed_text])

    return tfidf_feature

# Streamlit app layout
st.title("IMDB Movie Review Sentiment Analysis")
st.markdown("Enter a movie review below to get its sentiment prediction.")

# Text input area
user_input = st.text_area("", "Type your review here...")

# Predict button
if st.button("Predict Sentiment"):
    if user_input.strip() == "Type your review here..." or not user_input.strip():
        st.warning("Please enter a review to get a prediction.")
    else:
        # Preprocess the user's input
        preprocessed_input = preprocess_review(user_input, tfidf_vectorizer, lemmatizer, stop_words)

        # Make prediction
        prediction = best_model.predict(preprocessed_input)
        prediction_proba = best_model.predict_proba(preprocessed_input)

        st.subheader("Prediction:")
        if prediction[0] == 'positive':
            st.success(f"The sentiment is: **Positive**")
            st.write(f"Confidence: {prediction_proba[0][1]*100:.2f}%")
        else:
            st.error(f"The sentiment is: **Negative**")
            st.write(f"Confidence: {prediction_proba[0][0]*100:.2f}%")
