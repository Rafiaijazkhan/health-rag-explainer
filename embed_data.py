import json
import chromadb
from sentence_transformers import SentenceTransformer

# Load your cleaned data
with open("data/health_topics.json", "r", encoding="utf-8") as f:
    topics = json.load(f)

print(f"Loaded {len(topics)} topics.")

# Load the embedding model (this downloads it the first time, ~80MB)
print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# Set up ChromaDB - this creates a local folder called chroma_db
client = chromadb.PersistentClient(path="chroma_db")

# Create (or get) a collection - think of this like a "table"
# Delete old collection if it exists, so we start fresh each time
try:
    client.delete_collection(name="health_topics")
except:
    pass
collection = client.get_or_create_collection(name="health_topics")

# Add each topic to the collection
for i, item in enumerate(topics):
    embedding = model.encode(item["summary"]).tolist()

    collection.add(
        ids=[str(i)],
        embeddings=[embedding],
        documents=[item["summary"]],
        metadatas=[{
            "topic": item["topic"],
            "source": item["source"],
            "url": item["url"]
        }]
    )
    print(f"Added: {item['topic']}")

print(f"\nDone! Stored {len(topics)} topics in ChromaDB.")