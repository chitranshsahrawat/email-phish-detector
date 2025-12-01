# app.py (Streamlit app) - copy this file into your repo root
import streamlit as st
import joblib
import re
import numpy as np
import os

@st.cache_resource
def load_model():
    tf = joblib.load("model_artifacts/tfidf_vectorizer.joblib")
    clf = joblib.load("model_artifacts/logreg_clf.joblib")
    return tf, clf

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'http\S+', ' <URL> ', text)
    text = re.sub(r'[^a-z0-9<> ]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def top_contributors(text, tf, clf, top_n=8):
    vec = tf.transform([text])
    feats = tf.get_feature_names_out()
    coefs = clf.coef_[0]
    indices = vec.nonzero()[1]
    contribs = []
    for i in indices:
        contribs.append((feats[i], float(vec[0,i]) * float(coefs[i])))
    contribs_sorted = sorted(contribs, key=lambda x: x[1], reverse=True)
    return contribs_sorted[:top_n], contribs_sorted[-top_n:]

st.set_page_config(page_title="Email Phishing Detector", layout="centered")
st.title("📧 Email Phishing Detector (Toy Model)")

st.markdown("""
Enter the **subject + body** of an email. The model (TF-IDF + LogisticRegression) will predict probability of phishing.
""")

tf, clf = load_model()

text_input = st.text_area("Paste email (subject + body):", height=250)

if st.button("Check Email"):
    if not text_input.strip():
        st.warning("Please paste email text first.")
    else:
        cleaned = clean_text(text_input)
        x = tf.transform([cleaned])
        prob = clf.predict_proba(x)[0,1]
        pred = clf.predict(x)[0]
        label = "PHISHING ⚠️" if pred==1 else "LEGIT ✅"
        st.markdown(f"### Prediction: **{label}**")
        st.markdown(f"**Phishing probability:** {prob:.3f}")
        st.write("---")
        st.subheader("Top contributing tokens")
        pos, neg = top_contributors(cleaned, tf, clf, top_n=10)
        if pos:
            st.write("Top positive (towards phishing):")
            for w,score in pos:
                st.write(f"- **{w}** → {score:.4f}")
        if neg:
            st.write("Top negative (towards legit):")
            for w,score in neg:
                st.write(f"- **{w}** → {score:.4f}")

st.info("Note: This is a toy model for demo. For production, use a larger dataset and more robust checks (header analysis, URL scanning, SPF/DKIM checks).")
