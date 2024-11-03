import streamlit as st
import plotly.express as px
import pandas as pd
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from bs4 import BeautifulSoup
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import requests
import numpy as np
import os

# Set OpenAI API key from Streamlit secrets
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# Embedding model for similarity search
embedding_model = SentenceTransformer('paraphrase-MiniLM-L3-v2')

# Define candidate URLs (only examples, you would need to add more for other states)
candidate_urls = {
    'Pennsylvania': {
        'Dave McCormick (R)': 'https://www.davemccormickpa.com/issues/',
        'Bob Casey Jr. (D)': 'https://bobcasey.com/issues/'
    },
    # Add more states and candidates here
}

# Data setup
senate_race_states = list(candidate_urls.keys())  # Only states with defined candidates

state_data = pd.DataFrame({
    'state': senate_race_states,
    'abbrev': ['PA']  # Abbreviations for states; expand as needed
})
state_data['senate_race'] = '2024 Race'

# Display map
st.title("2024 U.S. Senate Races")
fig = px.choropleth(
    state_data,
    locations="abbrev",
    locationmode="USA-states",
    scope="usa",
    color="senate_race",
    color_discrete_map={'2024 Race': '#FF6347'},
    hover_name="state",
    title="2024 Senate Races by State"
)
st.plotly_chart(fig)

# State selection and link
st.write("### Select a State by Abbreviation")
clicked_state = st.selectbox("Select a state:", state_data['state'])

if clicked_state in candidate_urls:
    st.subheader(f"{clicked_state} Senate Race Chatbot")
    candidates = candidate_urls[clicked_state]
    politician = st.radio("Choose a Candidate:", list(candidates.keys()))
    url = candidates[politician]

    # Scrape and embed function
    def scrape_and_embed(url):
        response = requests.get(url)
        if response.status_code != 200:
            return None, None
        soup = BeautifulSoup(response.text, 'html.parser')
        filtered_text = []
        unwanted_tags = ["script", "style", "aside", "footer"]
        unwanted_classes = ["popup", "banner", "donation", "footer", "aside"]
        unwanted_keywords = ["donate", "support", "contribute", "subscribe"]
        
        for element in soup.find_all(True):
            if element.name in unwanted_tags or any(cls in element.get("class", []) for cls in element.get("class", [])):
                continue
            element_text = element.get_text().strip()
            if any(keyword in element_text.lower() for keyword in unwanted_keywords):
                continue
            if len(element_text) >= 50:
                filtered_text.append(element_text)
        
        text_sections = filtered_text
        embeddings = embedding_model.encode(text_sections)
        
        return text_sections, embeddings

    text_sections, embeddings = scrape_and_embed(url)
    steampunk_mode = st.checkbox("Enable Steampunk Mode")

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

    def retrieve_relevant_text(user_question, text_sections, embeddings):
        question_embedding = embedding_model.encode([user_question])
        similarities = cosine_similarity(question_embedding, embeddings)[0]
        most_similar_indices = np.argsort(similarities)[-3:]
        most_relevant_text = "\n\n".join([text_sections[i] for i in reversed(most_similar_indices)])
        return most_relevant_text

    if text_sections and embeddings.size > 0:
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

            context_text = retrieve_relevant_text(prompt_text, text_sections, embeddings)
            combined_input = f"Context:\n{context_text}\n\nUser Question:\n{prompt_text}"
            response = conversation_chain.run({"user_input": combined_input})
            
            st.session_state.messages.append({"role": "assistant", "content": response})
            with st.chat_message("assistant"):
                st.markdown(response)
    else:
        st.error("Failed to load website data into memory.")
else:
    st.write(f"No chatbot available for {clicked_state} Senate race yet.")
