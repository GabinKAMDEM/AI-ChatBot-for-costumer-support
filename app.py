import streamlit as st
import pandas as pd
import openai
import numpy as np
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings

# --- CONFIG ---
load_dotenv()
EMBEDDING_MODEL = 'paraphrase-MiniLM-L3-v2'
OPENAI_MODEL = 'gpt-4o-mini'
CSV_PATH = 'support_data.csv' 

@st.cache_data
# --- load data ---
def load_data(csv_path):
    df = pd.read_csv(csv_path)
    assert 'Question' in df.columns and 'Answer' in df.columns, "Le CSV doit contenir les colonnes 'Question' et 'Answer'"
    return df

dataset = load_data(CSV_PATH)
FEW_SHOT_EXAMPLES = [
    {"question": "I can't talk with a human agent", "answer": "Thank you for reaching out! I'm sorry to hear that you're facing difficulties in connecting with a human agent. Please rest assured that I'm here to assist you and resolve this issue. Could you please provide me with some more details about the problem you're experiencing? This will help me find the best solution for you."},
    {"question": "I have got to locate hte bills from {{Person Name}}", "answer": "I'll get right on it! I understand your need to locate the bills associated with {{Person Name}}. To assist you in finding the relevant invoices, could you please provide me with some additional details such as the date range or any other specific information that might help narrow down the search? Rest assured, I'm here to guide you every step of the way and ensure a seamless experience."},
    {"question": "I cannot pay, help me to inform of a problem with payments", "answer": "I apologize for the inconvenience caused by the payment issue you're experiencing. I understand how frustrating it can be when you encounter difficulties while trying to make a payment. Rest assured, I'm here to assist you in resolving this problem. To help us further investigate and address the issue, could you please provide more details about the problem you are encountering? Any specific error messages or steps leading up to the issue would be helpful. Your input is valuable, and we're committed to finding a solution together. Thank you for bringing this to our attention, and we'll work diligently to get it sorted."},
    {"question": "I want help speaking to customer service", "answer": "Grateful for your contact! I get the sense that you're looking for assistance in speaking to our customer service team. I'm here to guide you through the process and ensure that you have a seamless experience. To connect with our customer service, you can reach them during {{Customer Support Hours}}. If you have any other questions, concerns, or experiences you'd like to share, please feel free to let me know. I'm here to help!"},
    {"question": "I try to see th accepted payment options", "answer": "Thank you for reaching out! I completely understand that you tried to access the accepted payment options on our platform. I apologize for any inconvenience caused. Allow me to assist you in viewing our available payment methods. To check the accepted payment options, you can visit our website's \"Payment\" or \"Checkout\" page. There, you will find a comprehensive list of the various payment methods we accept. If you encounter any difficulties or have further questions, please don't hesitate to let me know. Your satisfaction is our priority, and I'm here to ensure a seamless payment experience for you."},
]


# --- VECTORSTORE & RAG ---
vectorstore = Chroma.from_texts(
    texts=dataset['Question'].tolist(),
    embedding=SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL),
    metadatas=[{"answer": a} for a in dataset['Answer'].tolist()]
)

def openai_llm(prompt):
    openai.api_key = os.getenv("OPENAI_API_KEY")
    try:
        response = openai.ChatCompletion.create(
            model=OPENAI_MODEL,
            messages=[{"role": "system", "content": "you are a customer support assistant"},
                      {"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Erreur OpenAI : {e}"
class CustomRAG:
    def __init__(self, vectorstore, llm, examples=None):
        self.vectorstore = vectorstore
        self.llm = llm 
        self.retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        self.examples = examples or []

    def format_prompt(self, question, contexts):
        few_shot = "".join([
            f"Q:{ex['question']}\nA: {ex['answer']}\n\n"
            for ex in self.examples
        ])
        context_text = "\n\n".join([
            f"{doc.metadata.get('answer', 'N/A')}"
            for i, doc in enumerate(contexts)
        ])
        prompt = (
            "You are a friendly, empathetic customer-support assistant who always uses polite, human-like language "
            "and addresses the user with respect.\n"
            "Give answers in <3 sentences and ask for clarification if needed.\n\n"
            f"Examples:\n{few_shot}\n"
            f"Contexts:\n{context_text}\n\n"
            f"Question: {question}\n"
            "Answer:\n"
        )
        return prompt

    def query(self, question):
        try:
            relevant_docs = self.retriever.get_relevant_documents(question)
            for doc in relevant_docs:
                if doc.metadata.get('answer'):
                    if self.is_relevant(question, doc.page_content):
                        return doc.metadata['answer']
            prompt = self.format_prompt(question, relevant_docs)
            response = self.llm(prompt)
            return self.clean_response(response)
        except Exception as e:
            return f"Erreur CustomRAG : {str(e)}"

    def is_relevant(self, question, context):
        question_words = set(question.lower().split())
        context_words = set(context.lower().split())
        overlap = len(question_words & context_words)
        return overlap / len(question_words) >= 0.3

    def clean_response(self, response):
        if isinstance(response, list):
            response = response[0] if response else ""
        lines = response.split('\n')
        cleaned_lines = []
        seen = set()
        for line in lines:
            line = line.strip()
            if line and line not in seen:
                cleaned_lines.append(line)
                seen.add(line)
        return '\n'.join(cleaned_lines) 
    
rag = CustomRAG(vectorstore, openai_llm, FEW_SHOT_EXAMPLES)

# --- UI STREAMLIT ---
st.title("🤖 Chatbot IA - Customer Support")
st.write("Ask your question, the chatbot will answer you !")

if 'history' not in st.session_state:
    st.session_state['history'] = []

user_input = st.text_input("Your question :", key="user_input")

if st.button("Send") and user_input.strip():
    answer = rag.query(user_input)
    st.session_state['history'].append((user_input, answer))

# Affichage de l'historique
tab1, tab2 = st.tabs(["Chat", "History"])
with tab1:
    if st.session_state['history']:
        for q, a in reversed(st.session_state['history']):
            st.markdown(f"**You :** {q}")
            st.markdown(f"**Bot :** {a}")
    else:
        st.info("Ask a question to start !")
with tab2:
    st.dataframe(dataset)

