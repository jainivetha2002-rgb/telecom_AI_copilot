# 🤖 AI Customer Support Copilot – ZENDS Communications

## 📌 Project Overview
ZENDS Communications is a virtual telecom company providing mobile, broadband, cloud, and IoT services.  
This project implements an **AI Customer Support Copilot** that automatically understands customer queries, detects intent and sentiment, retrieves relevant company policies using **Retrieval-Augmented Generation (RAG)**, and generates accurate, policy-compliant responses.

The system simulates a real-world **enterprise AI customer support solution** using **Hugging Face Transformers**, **NLP**, and **Streamlit**.

---

## 🎯 Problem Statement
Customer support teams in telecom companies handle a large volume of unstructured customer queries related to billing, refunds, technical issues, complaints, and product inquiries.  
Manual handling is slow, inconsistent, and resource-intensive.

This project solves the problem by:
- Automating customer query understanding
- Prioritizing negative or urgent queries
- Providing consistent, policy-based AI responses
- Demonstrating an end-to-end enterprise AI workflow

---

## 🧠 Approach Implemented (Fully Covered)

### 1. Synthetic Data Generation
Implemented using:
- **Intent Templates** (Billing, Refund, Technical, Complaint, Product)
- **Entity Injection** (plans, prices, countries)
- **Sentiment Variation** (positive, neutral, negative)
- **Paraphrasing using Hugging Face Transformers**

➡ Implemented in: `synthetic_data_generator.py`

---

### 2. Intent Classification
- Uses **Hugging Face Transformer model (DistilBERT)**
- Detects the purpose of customer queries
- Enables correct routing of queries

---

### 3. Sentiment Analysis
- Uses **Hugging Face Transformer model (RoBERTa)**
- Identifies customer emotion (negative, neutral, positive)
- Helps prioritize urgent or angry customer queries

---

### 4. Retrieval-Augmented Generation (RAG)
- Company policies are stored as documents
- Policies are vectorized using **TF-IDF**
- Relevant policy is retrieved using **cosine similarity**
- AI response is generated based on retrieved context

---

### 5. Streamlit Dashboard
- Interactive and colorful UI
- Real-time customer query input
- Displays:
  - Intent
  - Sentiment
  - Priority
  - Retrieved policy
  - AI suggested response

---

## 🛠️ Tech Stack
- **Python**
- **Hugging Face Transformers**
- **Natural Language Processing (NLP)**
- **Machine Learning**
- **Retrieval-Augmented Generation (RAG)**
- **Scikit-learn**
- **Streamlit**

---


