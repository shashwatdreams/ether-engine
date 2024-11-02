from flask import Flask, render_template, request, jsonify
from langchain.llms import OpenAI
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from bs4 import BeautifulSoup
import requests
import os

# Initialize Flask app
app = Flask(__name__)

# Set up OpenAI API key
os.environ["OPENAI_API_KEY"] = "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"

# Scrape website function
url = "https://www.donaldjtrump.com/issues"

def scrape_website(url):
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        text_content = soup.get_text()
        return text_content.strip()
    else:
        print(f"Failed to retrieve content: Status {response.status_code}")
        return None

# Initialize LangChain components
scraped_text = scrape_website(url)
if scraped_text:
    memory = ConversationBufferMemory()
    memory.save_context({"input": "Initial context"}, {"output": scraped_text})
    from langchain.chat_models import ChatOpenAI
    llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini")
    conversation_chain = ConversationChain(
        llm=llm,
        memory=memory,
        verbose=True
    )

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message")
    response = conversation_chain.run(user_input)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)
