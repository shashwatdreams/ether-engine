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
    'Arizona': {
        'Kari Lake (R)': 'https://karilake.com/issues/',
        'Ruben Gallego (D)': 'https://gallegoforarizona.com/issues/'
    },
    'California': {
        'Adam Schiff (D)': 'https://www.adamschiff.com/plans/',
        'Steve Garvey (R)': 'https://stevegarvey.com/steves-vision/'
    },
    'Connecticut': {
        'Chris Murphy (D)': 'https://chrismurphy.com/issues/',
        'Matthew Corey (R)': 'https://coreyforsenate.com/on-the-issues'
    },
    'Delaware': {
        'Lisa Blunt Rochester (D)': 'https://lisabluntrochester.com/issues/',
        'Eric Hansen (R)': 'https://voteerichansen.com/#Policies'
    },
    'Florida': {
        'Rick Scott (R)': 'https://rescueamerica.com/12-point-plan/',
        'Debbie Mucarsel-Powell (D)': 'https://debbiemucarselpowell.com/issues/'
    },
    'Hawaii': {
        'Mazie Hirono (D)': 'https://mazieforhawaii.com/issues/',
        'Bob McDermott (R)': 'https://votemcdermott.com/key-issues/'
    },
    'Indiana': {
        'Jim Banks (R)': 'https://banksforsenate.com/issues/',
        'Valerie McCray (D)': 'https://valeriemccray.org/platform'
    },
    'Maine': {
        'Angus King (I)': 'https://angusformaine.com/priorities/',
        'Demi Kouzounas (R)': 'https://demiforsenate.com/issues/'
    },
    'Maryland': {
        'Angela Alsobrooks (D)': 'https://www.angelaalsobrooks.com/priorities',
        'Larry Hogan (R)': 'https://larryhogan.com/strong-independent-leadership/'
    },
    'Massachusetts': {
        'Elizabeth Warren (D)': 'https://elizabethwarren.com/plans',
        'John Deaton (R)': 'https://johndeatonforsenate.com/issues/'
    },
    'Michigan': {
        'Elissa Slotkin (D)': 'https://elissaslotkin.org/priorities/',
        'Mike Rogers (R)': 'https://rogersforsenate.com/issues/'
    },
    'Minnesota': {
        'Amy Klobuchar (D)': 'https://amyklobuchar.com/issues/',
        'Royce White (R)': 'https://roycewhite.us/issues/'
    },
    'Mississippi': {
        'Roger Wicker (R)': 'https://www.wicker.senate.gov/biography',
        'Ty Pinkins (D)': 'https://www.typinkins.com/issues'
    },
    'Missouri': {
        'Josh Hawley (R)': 'https://joshhawley.com/',
        'Lucas Kunce (D)': 'https://lucaskunce.com/meet-lucas-kunce/'
    },
    'Montana': {
        'Jon Tester (D)': 'https://jontester.com/issues/',
        'Tim Sheehy (R)': 'https://timformt.com/on-the-issues/'
    },
    'Nebraska': {
        'Deb Fischer (R)': 'https://debfornebraska.com/meet-deb/',
        'Dan Osborn (I)': 'https://osbornforsenate.com/platform/'
    },
    'Nevada': {
        'Jacky Rosen (D)': 'https://rosenfornevada.com/issues/',
        'Sam Brown (R)': 'https://captainsambrown.com/issues/'
    },
    'New Jersey': {
        'Andy Kim (D)': 'https://www.andykim.com/issues/',
        'Curtis Bashaw (R)': 'https://curtisbashawforsenate.com/issues/'
    },
    'New Mexico': {
        'Martin Heinrich (D)': 'https://martinheinrich.com/issues/',
        'Nella Domenici (R)': 'https://www.nellaforsenate.com/issues'
    },
    'New York': {
        'Kirsten Gillibrand (D)': 'https://www.gillibrand.senate.gov/priorities/',
        'Mike Sapraicone (R)': 'https://mikesapraiconeforsenate.com/issues/'
    },
    'Ohio': {
        'Sherrod Brown (D)': 'https://sherrodbrown.com/issues/',
        'Bernie Moreno (R)': 'https://berniemoreno.com/why-im-running/'
    },
    'Pennsylvania': {
        'Bob Casey Jr. (D)': 'https://bobcasey.com/issues/',
        'Dave McCormick (R)': 'https://www.davemccormickpa.com/issues/'
    },
    'Rhode Island': {
        'Sheldon Whitehouse (D)': 'https://whitehouseforsenate.com/issues/',
        'Patricia Morgan (R)': 'https://www.patriciamorgan.com/on-the-issues-1'
    },
    'Tennessee': {
        'Marsha Blackburn (R)': 'https://www.blackburn.senate.gov/issues',
        'Gloria Johnson (D)': 'https://www.votegloriajohnson.com/meet-gloria/'
    },
    'Texas': {
        'Ted Cruz (R)': 'https://www.tedcruz.org/proven-record/',
        'Colin Allred (D)': 'https://colinallred.com/on-the-issues/'
    },
    'Utah': {
        'John Curtis (R)': 'https://www.johncurtis.org/issues/',
        'Caroline Gleich (D)': 'https://www.carolineforutah.com/issues'
    },
    'Vermont': {
        'Bernie Sanders (I)': 'https://berniesanders.com/issues/',
        'Gerald Malloy (R)': 'https://www.deploymalloy.com/landing/positions-24/'
    },
    'Virginia': {
        'Tim Kaine (D)': 'https://timkaine.com/issues/',
        'Hung Cao (R)': 'https://www.hungforva.com/'
    },
    'Washington': {
        'Maria Cantwell (D)': 'https://cantwell.com/issues/',
        'Raul Garcia (R)': 'https://garciaforwa.com/addressing-the-issues/'
    },
    'West Virginia': {
        'Jim Justice (R)': 'https://jimjusticewv.com/issues/',
        'Glenn Elliot (D)': 'https://www.elliottforwv.com/issues/'
    },
    'Wisconsin': {
        'Tammy Baldwin (D)': 'https://tammybaldwin.com/issues/',
        'Eric Hovde (R)': 'https://erichovde.com/issues/'
    },
    'Wyoming': {
        'John Barrasso (R)': 'https://barrassoforwyoming.com/#about',
        'Scott Morrow (D)': 'https://morrowforwyoming.com/'
    }
}

# Data setup
senate_race_states = list(candidate_urls.keys())  # Only states with defined candidates

state_abbrevs = {
    'Arizona': 'AZ', 'California': 'CA', 'Connecticut': 'CT', 'Delaware': 'DE',
    'Florida': 'FL', 'Hawaii': 'HI', 'Indiana': 'IN', 'Maine': 'ME', 'Maryland': 'MD',
    'Massachusetts': 'MA', 'Michigan': 'MI', 'Minnesota': 'MN', 'Mississippi': 'MS',
    'Missouri': 'MO', 'Montana': 'MT', 'Nebraska': 'NE', 'Nevada': 'NV', 'New Jersey': 'NJ',
    'New Mexico': 'NM', 'New York': 'NY', 'Ohio': 'OH', 'Pennsylvania': 'PA',
    'Rhode Island': 'RI', 'Tennessee': 'TN', 'Texas': 'TX', 'Utah': 'UT', 'Vermont': 'VT',
    'Virginia': 'VA', 'Washington': 'WA', 'West Virginia': 'WV', 'Wisconsin': 'WI', 'Wyoming': 'WY'
}

# Create DataFrame using the state_abbrevs dictionary
state_data = pd.DataFrame({
    'state': senate_race_states,
    'abbrev': [state_abbrevs[state] for state in senate_race_states],
    'senate_race': '2024 Race'
})

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
        Context:
        {user_input}

        Assistant:"""
    else:
        prompt_template = """
        You are a knowledgeable assistant trained on the information from a website. 
        Answer questions based on the website content as accurately as possible.
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
