import chromadb
from sentence_transformers import SentenceTransformer

# Load the same embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Connect to your existing ChromaDB
client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_or_create_collection(name="health_topics")

# Try a test question
query = "What causes bad headaches?"
query_embedding = model.encode(query).tolist()

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=3  # top 3 matches
)

print(f"Query: {query}\n")
for i in range(len(results["documents"][0])):
    topic = results["metadatas"][0][i]["topic"]
    distance = results["distances"][0][i]
    print(f"{i+1}. {topic}  (distance: {distance:.4f})")