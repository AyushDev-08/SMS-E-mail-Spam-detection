# Spam Message Detector

A machine learning project that classifies SMS messages as spam or ham (not spam), comparing a deep learning approach (Bidirectional LSTM) against a classical ML approach (Random Forest with TF-IDF). Includes a Streamlit web app to test both models side by side.

## Demo

Run the app locally and enter any message to see how both models classify it, along with their confidence scores.

```bash
streamlit run app.py
```

## Dataset

[UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) — 5,572 SMS messages labeled as spam or ham. The dataset is imbalanced (~13% spam), so it was balanced via downsampling before training.

## Approach

Preprocessing
    Removed punctuation
    Lowercased and removed English stopwords (NLTK)
    Downsampled the majority (ham) class to match spam count

Model 1 — Bidirectional LSTM (Deep Learning)
    Tokenized + padded sequences (max length 50)
    Embedding (64-dim) → Bidirectional LSTM (32 units) → Dropout → Dense → Sigmoid output
    Trained with EarlyStopping and ReduceLROnPlateau

Model 2 — Random Forest (Classical ML)
    TF-IDF vectorization (top 5,000 features)
    RandomForestClassifier with default trees

## Results

Bidirectional LSTM achieved the highest accuracy ~94%
Random Forest achieved competitive performance ~92.6%

Both models perform similarly, which suggests spam detection on short SMS text is largely a keyword-driven problem — classical ML with TF-IDF gets most of the way there, and the LSTM's ability to model word order/context adds only a modest improvement.

Random Forest also offers feature importance, showing which words most strongly drive spam predictions (e.g. "free", "win", "txt", "claim") — something the LSTM doesn't expose directly.

## Tech Stack

- Python 3.11
- TensorFlow / Keras (Bidirectional LSTM)
- Scikit-learn (Random Forest, TF-IDF, metrics)
- NLTK (stopword removal)
- Streamlit (web UI)
- Pandas, NumPy, Matplotlib, Seaborn, WordCloud

## Project Structure

```text
SMS-E-mail-Spam-detection/
│
├── data/
│   └── SMSSpamCollection
│
├── models/
│   ├── lstm_model.keras
│   ├── tokenizer.pkl
│   ├── random_forest.pkl
│   └── tfidf_vectorizer.pkl
│
├── src/
│   ├── lstm.py
│   ├── random_forest.py
│   ├── preprocessing.py
│   └── evaluate.py
│
├── screenshots/
│   ├── app_home.png      # Main Streamlit interface
│   ├── case1.png         # Example where models disagree
│   ├── spam.png          # Example spam prediction
│   └── ham.png           # Example legitimate message prediction
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
└── LICENSE
```

## Setup

### 1. Create and activate a virtual environment

```bash
python -m venv spamenv

# Windows
spamenv\Scripts\activate

# Linux / macOS
source spamenv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the models

```bash
python src/lstm.py
python src/random_forest.py
```

### 4. Launch the Streamlit application

```bash
streamlit run app.py
```

## Limitations

Trained on short, informal SMS-style text — accuracy is not validated on full-length emails, which differ in structure (headers, signatures, longer bodies)
Vocabulary limited to words seen during training (~4,000 unique words)

## Future Improvements

- Train on an email-specific dataset for broader generalization
- Handle email structure separately (subject line vs. body)
- Add more classical baselines (Naive Bayes, SVM, Logistic Regression) for a fuller comparison

## What I Learned

- Handling class imbalance through downsampling
- Building a full NLP preprocessing pipeline from scratch
- Tokenization/padding for sequence models vs. TF-IDF for classical ML
- Comparing a deep learning model against a classical ML baseline on the same problem
- Managing Python environments (conda vs venv) and resolving TensorFlow/Keras version conflicts

## Observation

Both the models were trained on the data but the way of training them is very different. This can cause the models to sometimes give different results.For Example: Promotional/marketing messages (e.g. "use code X for Y% off") sit in a gray zone the models weren't explicitly trained to handle — they're not malicious spam, but use spam-like language. LSTM tends to flag these as spam due to phrase patterns, while Random Forest's confidence drops near 50% (uncertain) since individual words appear in both classes.
