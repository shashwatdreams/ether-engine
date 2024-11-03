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

# Apply Steampunk theme styling if enabled
if steampunk_mode:
    st.markdown(
        """
        <style>
            /* Background styling for Steampunk Mode */
            body {
                background: #2E2A24 url('https://example.com/path-to-your-gears-background.jpg') no-repeat center center fixed;
                background-size: cover;
                color: #F3E5AB;
                font-family: 'Georgia', serif;
            }
            .stApp {
                background: rgba(46, 42, 36, 0.8);  /* Dark semi-transparent background */
                border-radius: 15px;
                padding: 20px;
            }
            /* Custom chat styling */
            .user-text {
                color: #FFD700;
                font-weight: bold;
                font-size: 1.2em;
                margin-bottom: 10px;
            }
            .assistant-text {
                color: #AFEEEE;
                font-size: 1em;
            }
            /* Sidebar and header adjustments */
            .stSidebar, .css-1d391kg {
                background-color: #4B3621;
                color: #DAA520;
            }
            /* Decorative gears */
            .gear {
                position: absolute;
                z-index: -1;
                opacity: 0.3;
            }
            .gear-1 { top: 20%; left: 10%; width: 100px; height: 100px; }
            .gear-2 { top: 50%; right: 10%; width: 80px; height: 80px; }
            .gear-3 { bottom: 20%; left: 30%; width: 120px; height: 120px; }
        </style>
        <div class="gear gear-1"><img src="https://example.com/gear1.png" width="100" height="100"></div>
        <div class="gear gear-2"><img src="https://example.com/gear2.png" width="80" height="80"></div>
        <div class="gear gear-3"><img src="https://example.com/gear3.png" width="120" height="120"></div>
        """,
        unsafe_allow_html=True
    )
    st.title("🔧 Steampunk Chatbot")
    st.write("Interact with the Victorian-themed chatbot and ask questions based on the website data.")
else:
    st.title("Website-Scraped Chatbot")
    st.write("Ask questions based on the scraped data from the website.")

if scraped_text:
    prompt = PromptTemplate(input_variables=["user_input"], template=prompt_template)
    
    memory = ConversationBufferMemory()
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
    conversation_chain = LLMChain(
        llm=llm,
        prompt=prompt,
        memory=memory
    )

    st.title("Election Policy Chatbot")
    st.write("Ask questions about each candidate's policy based directly off their website.")

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