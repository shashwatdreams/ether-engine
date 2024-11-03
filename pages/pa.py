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

embedding_model = SentenceTransformer('paraphrase-MiniLM-L3-v2')
st.title("PA Senate Race")
st.write("Select a candidate to analyze their policies directly from their website content.")

politician = st.radio("Choose a Candidate:", ["Dave McCormick (R)", "Bob Casey Jr. (D)"])
url = "https://www.davemccormickpa.com/issues/" if politician == "Dave McCormick (R)" else "https://bobcasey.com/issues/"

# Define a function to scrape and embed text
def scrape_and_embed(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
    }

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Check if the request was successful
    except requests.RequestException as e:
        st.error(f"Failed to retrieve website data: {e}")
        return None, None

    if not response.text.strip():
        st.error("Received empty response from the website.")
        return None, None

    st.write("Preview of raw HTML content:", response.text[:500])

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

    if not filtered_text:
        st.error("No relevant content found on the website after filtering.")
        return None, None

    embeddings = embedding_model.encode(filtered_text)
    return filtered_text, embeddings

text_sections, embeddings = scrape_and_embed(url)

if text_sections is None or embeddings is None:
    st.stop()  # Stop execution if data isn't retrieved

faiss_index = FAISS.from_texts(text_sections, OpenAIEmbeddings())

steampunk_mode = st.checkbox("Enable Steampunk Mode")

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

def retrieve_relevant_text(prompt_text, faiss_index):
    try:
        # Embed the prompt text
        question_embedding = OpenAIEmbeddings().embed_query(prompt_text)
        
        # Ensure embedding is a 2D array
        if isinstance(question_embedding, list):
            question_embedding = np.array(question_embedding)
        if len(question_embedding.shape) == 1:
            question_embedding = question_embedding.reshape(1, -1)
        
        # Check if dimensions match the FAISS index
        index_dim = faiss_index.index.d
        if question_embedding.shape[1] != index_dim:
            st.error(f"Embedding dimension {question_embedding.shape[1]} does not match FAISS index dimension {index_dim}.")
            return []

        # Perform similarity search
        docs = faiss_index.similarity_search_by_vector(question_embedding, k=3)
        return docs
    except AttributeError:
        st.error("Embedding method is not available. Check OpenAIEmbeddings setup.")
        return []
    except ValueError as e:
        st.error(f"Error during similarity search: {e}")
        return []

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
