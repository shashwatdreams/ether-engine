import streamlit as st
from langchain.llms import OpenAI
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from bs4 import BeautifulSoup
import requests
import os

# Set up the OpenAI API key
os.environ["OPENAI_API_KEY"] = "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"

# URL of the website to scrape
url = "https://www.donaldjtrump.com/issues"

# Function to scrape website content
def scrape_website(url):
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        text_content = soup.get_text()
        return text_content.strip()
    else:
        return None

# Scrape the website content
scraped_text = scrape_website(url)

# Initialize memory and conversation chain if scraping is successful
if scraped_text:
    memory = ConversationBufferMemory()
    memory.save_context({"input": "Initial context"}, {"output": scraped_text})

    # Initialize the language model and conversation chain
    llm = OpenAI(model="gpt-4o-mini", temperature=0)
    from langchain.chat_models import ChatOpenAI
    llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini")
    conversation_chain = ConversationChain(
        llm=llm,
        memory=memory,
        verbose=True
    )

    # Streamlit interface
    st.title("Website-Scraped Chatbot")
    st.write("Ask questions based on the scraped data from the website.")

    # Initialize session state for conversation history
    if "history" not in st.session_state:
        st.session_state.history = []

    # Display chat history
    for i, (user_msg, bot_msg) in enumerate(st.session_state.history):
        st.write(f"You: {user_msg}")
        st.write(f"Chatbot: {bot_msg}")

    # User input
    user_input = st.text_input("Type your message:")

    if user_input:
        # Get the chatbot response
        response = conversation_chain.run(user_input)

        # Update the conversation history
        st.session_state.history.append((user_input, response))

        # Display the new interaction
        st.write(f"You: {user_input}")
        st.write(f"Chatbot: {response}")
else:
    st.error("Failed to load website data into memory.")
