import streamlit as st
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="ZENDS AI Customer Support Copilot",
    page_icon="🤖",
    layout="wide"
)

# --------------------------------------------------
# CUSTOM CSS (COLOR + CARDS)
# --------------------------------------------------
st.markdown("""
<style>
.main-title {
    text-align: center;
    font-size: 42px;
    font-weight: 700;
    color: #1f4ed8;
}
.sub-title {
    text-align: center;
    font-size: 18px;
    color: #555;
    margin-bottom: 30px;
}
.card {
    padding: 20px;
    border-radius: 15px;
    background-color: #f8fafc;
    box-shadow: 0px 4px 10px rgba(0,0,0,0.08);
    height: 100%;
}
.intent {
    color: #2563eb;
    font-size: 24px;
    font-weight: bold;
}
.high {
    color: #dc2626;
    font-weight: bold;
}
.normal {
    color: #16a34a;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# TITLE SECTION
# --------------------------------------------------
st.markdown("<div class='main-title'>🤖 ZENDS AI Customer Support Copilot</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='sub-title'>AI-powered system using Hugging Face Transformers and RAG for telecom customer support</div>",
    unsafe_allow_html=True
)

# --------------------------------------------------
# LOAD MODELS
# --------------------------------------------------
@st.cache_resource
def load_models():
    intent_model = pipeline(
        "text-classification",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )
    sentiment_model = pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment"
    )
    return intent_model, sentiment_model

intent_classifier, sentiment_analyzer = load_models()

# --------------------------------------------------
# LOAD KNOWLEDGE BASE (RAG)
# --------------------------------------------------
with open("knowledge_base/zends_policies.txt", "r") as f:
    policies = f.read().split("\n\n")

vectorizer = TfidfVectorizer()
policy_vectors = vectorizer.fit_transform(policies)

def retrieve_policy(query):
    query_vec = vectorizer.transform([query])
    similarity = cosine_similarity(query_vec, policy_vectors)
    return policies[similarity.argmax()]

def generate_response(query):
    policy = retrieve_policy(query)
    return f"""{policy}

Our support team will assist you further if needed.
"""

# --------------------------------------------------
# CUSTOMER QUERY INPUT
# --------------------------------------------------
st.markdown("### 📝 Customer Query")

user_query = st.text_area(
    "",
    height=120,
    placeholder="Example: I was charged extra for my prepaid plan this month"
)

analyze = st.button("🚀 Analyze Customer Query")

# --------------------------------------------------
# PROCESS & DISPLAY RESULTS
# --------------------------------------------------
if analyze:
    if user_query.strip() == "":
        st.warning("Please enter a customer query.")
    else:
        intent = intent_classifier(user_query)[0]
        sentiment = sentiment_analyzer(user_query)[0]
        policy = retrieve_policy(user_query)
        response = generate_response(user_query)

        st.markdown("---")
        st.markdown("## 📊 AI Analysis")

        col1, col2, col3 = st.columns(3)

        # Intent Card
        with col1:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("### 🎯 Intent")
            st.markdown(f"<div class='intent'>{intent['label']}</div>", unsafe_allow_html=True)
            st.caption(f"Confidence: {intent['score']:.2f}")
            st.markdown("</div>", unsafe_allow_html=True)

        # Sentiment Card
        with col2:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("### 😊 Sentiment")
            if sentiment["label"] in ["LABEL_0", "NEGATIVE"]:
                st.error("Negative / Angry 😠")
            elif sentiment["label"] in ["LABEL_1", "NEUTRAL"]:
                st.warning("Neutral 😐")
            else:
                st.success("Positive 😊")
            st.caption(f"Confidence: {sentiment['score']:.2f}")
            st.markdown("</div>", unsafe_allow_html=True)

        # Priority Card
        with col3:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("### 🚦 Priority")
            if sentiment["label"] in ["LABEL_0", "NEGATIVE"]:
                st.markdown("<div class='high'>HIGH PRIORITY</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='normal'>NORMAL PRIORITY</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # RAG Policy
        st.markdown("---")
        st.markdown("## 📄 Retrieved Policy (RAG)")
        st.info(policy)

        # AI Response
        st.markdown("## 🤖 AI Suggested Response")
        st.success(response)

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:gray;'>ZENDS Communications | AI Customer Support Automation</p>",
    unsafe_allow_html=True
)
