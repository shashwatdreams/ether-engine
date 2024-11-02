import streamlit as st
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from bs4 import BeautifulSoup
import requests
import os

# Set up OpenAI API key
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# Define the website to scrape
url = "https://www.donaldjtrump.com/issues"
def scrape_website(url):
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        return soup.get_text().strip()
    else:
        return None

# Scrape content and set up memory if successful
scraped_text = scrape_website(url)
if scraped_text:
    # Define a prompt template that takes a single input
    prompt_template = """
    You are a knowledgeable assistant trained on the information from a website. 
    Answer questions based on the website content as accurately as possible.
    If you cannot find the answer in the provided content, politely indicate that.
    Context:
    {user_input}

    Assistant:"""
    
    prompt = PromptTemplate(input_variables=["user_input"], template=prompt_template)
    
    # Set up the language model and chain with memory
    memory = ConversationBufferMemory()
    llm = ChatOpenAI(model_name="gpt-4-turbo", temperature=0)
    conversation_chain = LLMChain(
        llm=llm,
        prompt=prompt,
        memory=memory
    )

    # Streamlit chat interface
    st.title("Website-Scraped Chatbot")
    st.write("Ask questions based on the scraped data from the website.")

    # Initialize chat history in session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User input and generating responses
    if prompt_text := st.chat_input("Enter your message here..."):
        st.session_state.messages.append({"role": "user", "content": prompt_text})
        with st.chat_message("user"):
            st.markdown(prompt_text)

        # Combine context and input into one user_input key
        combined_input = f"Context:\n{scraped_text}\n\nUser Question:\n{prompt_text}"
        response = conversation_chain.run({"user_input": combined_input})
        
        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)
else:
    st.error("Failed to load website data into memory.")
