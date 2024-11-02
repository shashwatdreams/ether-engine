from flask import Flask, render_template, request, jsonify
from langchain_community.llms import OpenAI
from langchain_openai import ChatOpenAI  # Updated import for ChatOpenAI
from bs4 import BeautifulSoup
import requests
import os

# Initialize Flask app
app = Flask(__name__)

# Set up OpenAI API key
os.environ["OPENAI_API_KEY"] = "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"

# Scrape website function
urls = [
    "https://www.donaldjtrump.com/issues",
    "https://kamalaharris.com/issues/"
]

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
llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini")

# Simple memory to keep track of conversation
conversation_history = []

# Scrape text from both websites and save the context
for url in urls:
    scraped_text = scrape_website(url)
    if scraped_text:
        conversation_history.append(f"Context: {scraped_text}")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message")
    
    # Append user input to the conversation history
    conversation_history.append(f"User: {user_input}")

    # Prepare the full conversation for LLM
    full_prompt = "\n".join(conversation_history) + "\nAI:"
    
    # Generate response from the model
    response = llm.generate(full_prompt)
    
    # Append AI response to the conversation history
    conversation_history.append(f"AI: {response}")
    
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)
