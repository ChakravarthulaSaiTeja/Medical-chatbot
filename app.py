from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from pinecone import Pinecone
from dotenv import load_dotenv
from src.prompt import *
import os

# Disable tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Initialize Flask app
app = Flask(__name__)

# Load environment variables
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

if not PINECONE_API_KEY or not OPENAI_API_KEY:
    raise ValueError("API keys are missing! Check your .env file.")

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

# Load Hugging Face embeddings
embeddings = download_hugging_face_embeddings()
if not embeddings:
    raise ValueError("Failed to load Hugging Face embeddings. Check API key and model.")

index_name = "medicalbot-1"

# Load existing Pinecone index
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name, embedding=embeddings
)

# Set up retriever
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Initialize OpenAI LLM
llm = OpenAI(temperature=0.4, max_tokens=500)

# Define prompt
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# Create LLM chains
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)


# Flask Routes
@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/get", methods=["POST"])
def chat():
    """Handles chatbot requests."""
    try:
        data = request.get_json() if request.is_json else request.form
        msg = data.get("msg", "").strip()

        if not msg:
            return jsonify({"error": "No input received"}), 400

        try:
            response = rag_chain.invoke({"input": msg})
            answer = response.get("answer", "Sorry, I couldn't understand that.")
        except Exception as e:
            print("Error in RAG Chain:", str(e))
            answer = "Sorry, an error occurred."

        return jsonify({"answer": answer})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
