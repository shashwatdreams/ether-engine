import streamlit as st
from langchain.llms import OpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from bs4 import BeautifulSoup
import requests
import os

os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

url = "https://www.donaldjtrump.com/issues"
def scrape_website(url):
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        return soup.get_text().strip()
    else:
        return None

scraped_text = scrape_website(url)
if scraped_text:
    memory = ConversationBufferMemory()
    memory.save_context({"input": "Initial context"}, {"output": scraped_text})

    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0) 
    conversation_chain = ConversationChain(
        llm=llm,
        memory=memory,
        verbose=True
    )

    st.title("Website-Scraped Chatbot")
    st.write("Ask questions based on the scraped data from the website.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Enter your message here..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        response = conversation_chain.run(prompt)
        st.session_state.messages.append({"role": "assistant", "content": response})

        with st.chat_message("assistant"):
            st.markdown(response)
else:
    st.error("Failed to load website data into memory.")