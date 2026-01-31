import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# -------------------- Load Dataset --------------------
df = pd.read_csv("spam.csv", encoding='latin-1')
df = df.iloc[:, :2]
df.columns = ['label', 'message']
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# -------------------- Train Model --------------------
X = df['message']
y = df['label']

vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = MultinomialNB()
model.fit(X_train, y_train)

# -------------------- Website UI --------------------
st.title("📧 Email Spam Classifier")
st.write("Enter a message to check whether it is Spam or Ham")

user_input = st.text_area("Enter your message here")

if st.button("Predict"):
    if user_input:
        text_vec = vectorizer.transform([user_input])
        prediction = model.predict(text_vec)

        if prediction[0] == 1:
            st.error("🚨 This is SPAM")
        else:
            st.success("✅ This is HAM (Not Spam)")
    else:
        st.warning("Please enter a message.")
