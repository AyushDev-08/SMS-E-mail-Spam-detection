import streamlit as st
import joblib
import string
import nltk
from nltk.corpus import stopwords
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences

nltk.download('stopwords', quiet=True)

# -------------------- Page Config --------------------
st.set_page_config(page_title="Spam Detector", page_icon="📧", layout="centered")

# -------------------- Load Models (cached so they load once) --------------------
@st.cache_resource
def load_models():
    lstm_model = tf.keras.models.load_model('models/spam_detector_lstm.keras')
    tokenizer = joblib.load('models/tokenizer.pkl')
    rf_model = joblib.load('models/random_forest.pkl')
    tfidf = joblib.load('models/tfidf_vectorizer.pkl')
    return lstm_model, tokenizer, rf_model, tfidf

lstm_model, tokenizer, rf_model, tfidf = load_models()

MAX_LEN = 50

# -------------------- Preprocessing (same as training) --------------------
def remove_punctuations(message):
    temp = str.maketrans('', '', string.punctuation)
    return message.translate(temp)

def remove_stopwords(message):
    stop_words = stopwords.words('english')
    imp_words = [w.lower() for w in str(message).split() if w.lower() not in stop_words]
    return " ".join(imp_words)

def clean_text(message):
    message = remove_punctuations(message)
    message = remove_stopwords(message)
    return message

# -------------------- Predictions --------------------
def predict_lstm(message):
    cleaned = clean_text(message)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding='post', truncating='post')
    prob = float(lstm_model.predict(padded, verbose=0)[0][0])
    return prob

def predict_rf(message):
    cleaned = clean_text(message)
    vec = tfidf.transform([cleaned])
    prob = float(rf_model.predict_proba(vec)[0][1])#gives probability estimates for class prediction
    return prob

# -------------------- UI --------------------
st.title("Spam Email Detector")
st.caption("Compares a Bidirectional LSTM (deep learning) against a Random Forest (classical ML).")

message = st.text_area("Enter a message to check:", height=140, placeholder="e.g. Congratulations! You've won a free prize, click here to claim now.")

col1, col2 = st.columns(2)
with col1:
    check = st.button("Check Message", use_container_width=True)
with col2:
    clear = st.button("Clear", use_container_width=True)

if clear:
    st.rerun()

if check:
    if not message.strip():
        st.warning("Please enter a message first.")
    else:
        lstm_prob = predict_lstm(message)
        rf_prob = predict_rf(message)

        st.divider()
        result_col1, result_col2 = st.columns(2)

        with result_col1:
            st.subheader("Bidirectional LSTM")
            label = "Spam" if lstm_prob > 0.5 else "Not Spam"
            confidence = lstm_prob if lstm_prob > 0.5 else 1 - lstm_prob
            if label == "Spam":
                st.error(f"**{label}**")
            else:
                st.success(f"**{label}**")
            st.progress(confidence)
            st.caption(f"{confidence*100:.1f}% confident")

        with result_col2:
            st.subheader("Random Forest")
            label = "Spam" if rf_prob > 0.5 else "Not Spam"
            confidence = rf_prob if rf_prob > 0.5 else 1 - rf_prob
            if label == "Spam":
                st.error(f"**{label}**")
            else:
                st.success(f"**{label}**")
            st.progress(confidence)
            st.caption(f"{confidence*100:.1f}% confident")

        st.divider()
        if lstm_prob > 0.5 and rf_prob > 0.5:
            st.error("Both models agree — this is likely Spam.")
        elif lstm_prob <= 0.5 and rf_prob <= 0.5:
            st.success("Both models agree — this is likely Not Spam.")
        else:
            st.warning("Models disagree — this message may be borderline or ambiguous.")
