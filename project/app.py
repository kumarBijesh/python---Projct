import pandas as pd
import re
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Define the text preprocessing function
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)

    # Remove HTML Tags
    text = re.sub(r'<.*?>', '', text)

    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)

    # Remove newline characters
    text = re.sub(r'\n', ' ', text)

    # Remove digits
    text = re.sub(r'\d', '', text)

    return text


# Load datasets and prepare the data
true = pd.read_csv('True.csv')
fake = pd.read_csv('Fake.csv')

# Add labels and concatenate datasets
true['label'] = 1  # Genuine news
fake['label'] = 0  # Fake news

news = pd.concat([fake, true], axis=0)
news = news.drop(['title', 'subject', 'date'], axis=1).sample(frac=1).reset_index(drop=True)

# Apply text preprocessing
news['text'] = news['text'].apply(preprocess_text)

# Split dataset into training and test sets
x = news['text']
y = news['label']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# Apply TF-IDF vectorization
vectorizer = TfidfVectorizer()
xv_train = vectorizer.fit_transform(x_train)
xv_test = vectorizer.transform(x_test)

# Train various models
models = {
    'Logistic Regression': LogisticRegression(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'Gradient Boosting': GradientBoostingClassifier(),
}

for model_name, model in models.items():
    model.fit(xv_train, y_train)

# Define a function to convert the prediction output to a human-readable label
def output_label(prediction):
    return "Fake News" if prediction == 0 else "Genuine News"

# Create the Streamlit app interface
st.title("Fake News Detection Model")

st.write("This app allows you to test news articles to determine whether they are fake or genuine.")

news_article = st.text_area("Enter the news article for testing")

if news_article:
    # Preprocess and transform the input
    processed_article = preprocess_text(news_article)
    transformed_article = vectorizer.transform([processed_article])

    # Display the predictions from different models
    st.write("Model Predictions:")
    for model_name, model in models.items():
        prediction = model.predict(transformed_article)[0]
        label = output_label(prediction)
        st.write(f"{model_name}: {label}")
