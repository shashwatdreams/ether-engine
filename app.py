import streamlit as st
import openai
from bs4 import BeautifulSoup
import requests

openai.api_key = st.secrets["OPENAI_API_KEY"]

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
  st.title("Website-Scraped Chatbot")
  st.write("Ask questions based on the scraped data from the website.")

  if "messages" not in st.session_state:
    st.session_state.messages = []

  for message in st.session_state.messages:
    with st.chat_message(message["role"]):
      st.markdown(message["content"])

  if prompt := st.chat_input("Enter   
 your message here..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
      st.markdown(prompt)

    try:
      # Use the recommended Completion.create function from v1.0.0 onwards
      response = openai.Completion.create(
          engine="text-davinci-003",  # Replace with your preferred model
          prompt="\n".join([message["content"] for message in st.session_state.messages] + [scraped_text]),
          max_tokens=1024,
          n=1,
          stop=None,
          temperature=0.7,
      ).choices[0].text

      st.session_state.messages.append({"role": "assistant", "content": response})
      with st.chat_message("assistant"):
        st.markdown(response)

    except Exception as e:
      st.error(f"An error occurred:   
 {e}")
else:
  st.error("Failed to load website data into memory.")