import streamlit as st
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
import requests
import numpy as np
import os

# Set up the OpenAI API key
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# Load embedding model
embedding_model = SentenceTransformer('paraphrase-MiniLM-L3-v2')
st.title("PA Senate Race")
st.write("Select a candidate to analyze their policies directly from their website content.")

# Choose the politician
politician = st.radio("Choose a Candidate:", ["Dave McCormick (R)", "Bob Casey Jr. (D)"])
url = "https://www.davemccormickpa.com/issues/" if politician == "Dave McCormick (R)" else "https://bobcasey.com/issues/"

# Define a function to scrape and embed text
def scrape_and_embed(url):
    response = requests.get(url)
    if response.status_code != 200:
        return None, None
    
    soup = BeautifulSoup(response.text, 'html.parser')
    filtered_text = []
    unwanted_tags = ["script", "style", "aside", "footer"]
    unwanted_keywords = ["donate", "support", "contribute", "subscribe"]
    
    for element in soup.find_all(True):
        if element.name in unwanted_tags or any(keyword in element.get_text().lower() for keyword in unwanted_keywords):
            continue
        element_text = element.get_text().strip()
        if len(element_text) >= 50:
            filtered_text.append(element_text)
    
    # Generate embeddings for each section of text
    text_sections = filtered_text
    embeddings = embedding_model.encode(text_sections)
    return text_sections, embeddings

# Scrape and embed content
text_sections, embeddings = scrape_and_embed(url)
if text_sections is not None and embeddings is not None:
    # Load the embeddings into FAISS vector store for retrieval
    faiss_index = FAISS.from_texts(text_sections, OpenAIEmbeddings())

steampunk_mode = st.checkbox("Enable Steampunk Mode")

# Define prompt templates for normal and steampunk modes
prompt_template = """
    You are a knowledgeable assistant trained on the information from a website. 
    Answer questions based on the website content as accurately as possible.
    If you cannot find the answer in the provided content, politely indicate that.
    Context:
    {user_input}
    
    Assistant:"""
if steampunk_mode:
    prompt_template = """
    Talk like you are in the Victorian time, in a steampunk theme. But keep your vocabulary minimal and make everything super understandable. 
    You are a knowledgeable assistant trained on the information from a website. 
    Answer questions based on the website content as accurately as possible.
    If you cannot find the answer in the provided content, politely indicate that.
    Context:
    {user_input}
    
    Assistant:"""

# Function to retrieve relevant context using FAISS
def retrieve_relevant_text(user_question, faiss_index):
    # Embed the user's question and retrieve similar content from FAISS
    question_embedding = embedding_model.encode([user_question])
    docs = faiss_index.similarity_search_by_vector(question_embedding, k=3)
    most_relevant_text = "\n\n".join([doc.page_content for doc in docs])
    return most_relevant_text

# Initialize LangChain prompt and chain
prompt = PromptTemplate(input_variables=["user_input"], template=prompt_template)
memory = ConversationBufferMemory()
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
conversation_chain = LLMChain(
    llm=llm,
    prompt=prompt,
    memory=memory
)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input and response generation
if prompt_text := st.chat_input("Enter your message here..."):
    st.session_state.messages.append({"role": "user", "content": prompt_text})
    with st.chat_message("user"):
        st.markdown(prompt_text)

    # Retrieve relevant text using FAISS index
    context_text = retrieve_relevant_text(prompt_text, faiss_index)
    combined_input = f"Context:\n{context_text}\n\nUser Question:\n{prompt_text}"
    response = conversation_chain.run({"user_input": combined_input})
    
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)
else:
    st.error("Failed to load website data into memory.")
