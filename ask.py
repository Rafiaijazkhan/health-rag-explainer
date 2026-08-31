import os
from dotenv import load_dotenv
import chromadb
from sentence_transformers import SentenceTransformer
from google import genai

# Load API key from .env
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# Set up Gemini client
client_ai = genai.Client(api_key=api_key)

# Set up embedding model + ChromaDB
model = SentenceTransformer("all-MiniLM-L6-v2")
client_db = chromadb.PersistentClient(path="chroma_db")
collection = client_db.get_or_create_collection(name="health_topics")

def ask_question(question):
    # Step 1: Retrieve relevant info
    query_embedding = model.encode(question).tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=2)

    retrieved_text = results["documents"][0][0]
    topic = results["metadatas"][0][0]["topic"]
    source = results["metadatas"][0][0]["source"]
    distance = results["distances"][0][0]

    # Step 2: Build the prompt with retrieved context
    prompt = f"""You are a helpful health information assistant. 
Answer the user's question using ONLY the information below. 
Explain it in simple, plain language. Always mention the source at the end.
Include a brief note to consult a doctor for personal medical advice.

Retrieved information about "{topic}":
{retrieved_text}

User's question: {question}

Answer:"""

    # Step 3: Send to Gemini
    response = client_ai.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt
    )

    answer = response.text

    print(f"\n--- Retrieved topic: {topic} (distance: {distance:.4f}) ---\n")
    print(answer)
    print(f"\nSource: {source}")

# Test it
if __name__ == "__main__":
    ask_question("What causes bad headaches?")