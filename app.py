import streamlit as st
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from bs4 import BeautifulSoup
import requests
import os
from langchain.embeddings import OpenAIEmbeddings
from chromadb import Client as ChromaClient
from chromadb.config import Settings

os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

st.title("Election Policy Chatbot")
st.write("Select a candidate to analyze their policies directly from their website content.")

politician = st.radio("Choose a Candidate:", ["Donald Trump", "Kamala Harris"])

url = "https://www.donaldjtrump.com/issues" if politician == "Donald Trump" else "https://kamalaharris.com/issues/"

steampunk_mode = st.checkbox("Enable Steampunk Mode")

def scrape_website_filtered(url):
    response = requests.get(url)
    if response.status_code != 200:
        return None

    soup = BeautifulSoup(response.text, 'html.parser')
    filtered_text = []

    unwanted_tags = ["script", "style", "aside", "footer"]
    unwanted_classes = ["popup", "banner", "donation", "footer", "aside"]
    unwanted_keywords = ["donate", "support", "contribute", "subscribe"]

    for element in soup.find_all(True):  # Finds all tags
        if element.name in unwanted_tags:
            continue
        
        if any(cls in element.get("class", []) for cls in unwanted_classes):
            continue

        element_text = element.get_text().strip()
        if any(keyword in element_text.lower() for keyword in unwanted_keywords):
            continue

        # Only keep substantial content
        if len(element_text) >= 50:
            filtered_text.append(element_text)

    return filtered_text

scraped_segments = scrape_website_filtered(url)

chroma_client = ChromaClient(Settings())
embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002")

if scraped_segments:
    collection_name = f"{politician.lower()}_policy_segments"
    collection = chroma_client.get_or_create_collection(collection_name, embedding_function=embedding_model.embed_text)

    for i, segment in enumerate(scraped_segments):
        collection.add_documents([{"text": segment, "metadata": {"index": i}}])

if steampunk_mode:
    prompt_template = """
    Talk like you are in the victorian time, in a steampunk theme. But keep your vocabulary minimal and make everything super understandable. 
    You are a knowledgeable assistant trained on the information from a website. 
    Answer questions based on the website content as accurately as possible.
    If you cannot find the answer in the provided content, politely indicate that.
    Context:
    {user_input}

    Assistant:"""
else:
    prompt_template = """
    You are a knowledgeable assistant trained on the information from a website. 
    Answer questions based on the website content as accurately as possible.
    If you cannot find the answer in the provided content, politely indicate that.
    Context:
    {user_input}

    Assistant:"""

def get_relevant_content(user_input):
    results = collection.query(query_texts=[user_input], n_results=3)
    relevant_segments = [result["text"] for result in results["documents"][0]]
    return "\n".join(relevant_segments)

if scraped_segments:
    prompt = PromptTemplate(input_variables=["user_input"], template=prompt_template)
    
    memory = ConversationBufferMemory()
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
    conversation_chain = LLMChain(
        llm=llm,
        prompt=prompt,
        memory=memory
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt_text := st.chat_input("Enter your message here..."):
        st.session_state.messages.append({"role": "user", "content": prompt_text})
        with st.chat_message("user"):
            st.markdown(prompt_text)
        relevant_content = get_relevant_content(prompt_text)
        combined_input = f"Context:\n{relevant_content}\n\nUser Question:\n{prompt_text}"
        response = conversation_chain.run({"user_input": combined_input})
        
        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)
else:
    st.error("Failed to load website data into memory.")
