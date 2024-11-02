import streamlit as st
from langchain.llms import OpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from bs4 import BeautifulSoup
import requests
import openai
import os

# Set up the OpenAI API key
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Scrape website content to load into LangChain memory
url = "https://www.donaldjtrump.com/issues"
def scrape_website(url):
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        return soup.get_text().strip()
    else:
        return None

# Load scraped content into memory if available
scraped_text = scrape_website(url)
if scraped_text:
    memory = ConversationBufferMemory()
    memory.save_context({"input": "Initial context"}, {"output": scraped_text})

    # Initialize the LangChain conversation with OpenAI model
    conversation_chain = ConversationChain(
        llm=OpenAI(model="gpt-3.5-turbo", temperature=0),
        memory=memory,
        verbose=True
    )

    # Streamlit chat interface
    st.title("Website-Scraped Chatbot")
    st.write("Ask questions based on the scraped data from the website.")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User input and response generation
    if prompt := st.chat_input("Enter your message here..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate chatbot response
        response = conversation_chain.run(prompt)
        st.session_state.messages.append({"role": "assistant", "content": response})

        with st.chat_message("assistant"):
            st.markdown(response)
else:
    st.error("Failed to load website data into memory.")
