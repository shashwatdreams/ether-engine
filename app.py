import streamlit as st
import openai
from bs4 import BeautifulSoup
import requests

# Set up the OpenAI API key
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Scrape website content
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

        # Generate chatbot response directly using OpenAI API
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": scraped_text},
                    *st.session_state.messages
                ]
            )
            answer = response['choices'][0]['message']['content']
            st.session_state.messages.append({"role": "assistant", "content": answer})

            with st.chat_message("assistant"):
                st.markdown(answer)
        except Exception as e:
            st.error(f"An error occurred: {e}")
else:
    st.error("Failed to load website data into memory.")