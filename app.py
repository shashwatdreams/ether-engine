import streamlit as st
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from bs4 import BeautifulSoup
import requests
import os

os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

url = "https://www.donaldjtrump.com/issues"

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

    return "\n".join(filtered_text)

scraped_text = scrape_website_filtered(url)

steampunk_mode = st.sidebar.checkbox("Steampunk Mode")

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



if scraped_text:
    prompt = PromptTemplate(input_variables=["user_input"], template=prompt_template)
    
    memory = ConversationBufferMemory()
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
    conversation_chain = LLMChain(
        llm=llm,
        prompt=prompt,
        memory=memory
    )

    st.title("Website-Scraped Chatbot")
    st.write("Ask questions based on the scraped data from the website.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt_text := st.chat_input("Enter your message here..."):
        st.session_state.messages.append({"role": "user", "content": prompt_text})
        with st.chat_message("user"):
            st.markdown(prompt_text)

        combined_input = f"Context:\n{scraped_text}\n\nUser Question:\n{prompt_text}"
        response = conversation_chain.run({"user_input": combined_input})
        
        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)
else:
    st.error("Failed to load website data into memory.")