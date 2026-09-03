import os
import time
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq

load_dotenv()

client_ai = Groq(api_key=os.getenv("GROQ_API_KEY"))

model = SentenceTransformer("all-MiniLM-L6-v2")
client_db = chromadb.PersistentClient(path="chroma_db")
collection = client_db.get_or_create_collection(name="health_topics")

app = Flask(__name__)

answer_cache = {}


def generate_with_retry(prompt, retries=2, delay=1):
    for attempt in range(retries + 1):
        try:
            response = client_ai.chat.completions.create(
                model="openai/gpt-oss-20b",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500
            )
            content = response.choices[0].message.content
            if content and content.strip():
                return content
            print(f"Groq returned empty content (attempt {attempt+1})")
        except Exception as e:
            print(f"Groq call failed (attempt {attempt+1}): {e}")
        if attempt < retries:
            time.sleep(delay)
    return None


def get_answer(question):
    cache_key = question.strip().lower()
    if cache_key in answer_cache:
        print(f"Cache hit for: {question}")
        return answer_cache[cache_key]

    t0 = time.time()
    query_embedding = model.encode(question).tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=3)
    t1 = time.time()
    print(f"Retrieval took {t1 - t0:.2f}s")

    best_distance = results["distances"][0][0]
    print(f"DEBUG - Question: '{question}' | Best match distance: {best_distance}")

    if best_distance > 1.3:
        result = {
            "topic": "No close match",
            "source": "N/A",
            "distance": round(best_distance, 4),
            "answer": "I don't have reliable information on this topic in my current database. This assistant only covers general health topics like headaches, diabetes, allergies, and similar conditions. Please consult a doctor or a trusted medical source for this question.",
            "related": []
        }
        return result

    retrieved_chunks = []
    topics_found = []
    for i in range(len(results["documents"][0])):
        retrieved_chunks.append(results["documents"][0][i])
        topics_found.append(results["metadatas"][0][i]["topic"])

    retrieved_text = "\n\n---\n\n".join(retrieved_chunks)
    topic = ", ".join(topics_found)
    source = results["metadatas"][0][0]["source"]
    distance = results["distances"][0][0]

    prompt = f"""You are a helpful health information assistant.
Carefully read the user's actual question below and answer exactly what they asked —
whether that's about causes, prevention, symptoms, treatment, or anything else.
Use ONLY the information provided below as your factual basis.
Explain it in simple, plain language. Always mention the source at the end.
Include a brief note to consult a doctor for personal medical advice.
Keep your answer under 80 words.

After your answer, on a new line, write exactly:
RELATED: question one? | question two? | question three?
Where each question is a short, natural follow-up someone might ask next about this same topic (based on the retrieved information), phrased simply, each ending with a question mark.

Retrieved information:
{retrieved_text}

User's question: {question}

Answer:"""

    t2 = time.time()
    answer_text = generate_with_retry(prompt)
    t3 = time.time()
    print(f"Groq call took {t3 - t2:.2f}s")

    if answer_text is None:
        result = {
            "topic": topic,
            "source": source,
            "distance": round(distance, 4),
            "answer": "The connection timed out while generating a response. Please try asking again.",
            "related": []
        }
        return result

    related_questions = []
    main_answer = answer_text

    if "RELATED:" in answer_text:
        parts = answer_text.split("RELATED:")
        main_answer = parts[0].strip()
        related_line = parts[1].strip()
        related_questions = [q.strip() for q in related_line.split("|") if q.strip()]

    result = {
        "topic": topic,
        "source": source,
        "distance": round(distance, 4),
        "answer": main_answer,
        "related": related_questions
    }
    answer_cache[cache_key] = result
    return result


def get_interaction(med1, med2):
    prompt = f"""You are a cautious health information assistant, not a doctor or pharmacist.
Explain in plain, simple language whether {med1} and {med2} are generally known to interact or are commonly considered safe to take together.
Give a short reasoning (2-4 sentences).
Do NOT give a dosing recommendation. Do NOT tell the user they are definitely safe.
Always end with a clear instruction to confirm with a pharmacist or doctor before combining any medications.
Keep your answer under 80 words.

Medicine 1: {med1}
Medicine 2: {med2}

Answer:"""

    answer_text = generate_with_retry(prompt)
    if answer_text is None:
        return "The connection timed out while generating a response. Please try again."
    return answer_text


def translate_text(text, target_language):
    prompt = f"""Translate the following health information into {target_language}.
Keep it simple and clear. Preserve the meaning exactly, don't add new information.

Text to translate:
{text}

Translation:"""

    translated_text = generate_with_retry(prompt)
    if translated_text is None:
        return "Translation timed out. Please try again."
    return translated_text


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    question = data.get("question", "").strip()

    if not question:
        return jsonify({"error": "Please enter a question."}), 400

    try:
        result = get_answer(question)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/interaction", methods=["POST"])
def interaction():
    data = request.get_json()
    med1 = data.get("med1", "").strip()
    med2 = data.get("med2", "").strip()

    if not med1 or not med2:
        return jsonify({"error": "Please enter both medicine names."}), 400

    try:
        result = get_interaction(med1, med2)
        return jsonify({"result": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/translate", methods=["POST"])
def translate():
    data = request.get_json()
    text = data.get("text", "").strip()
    language = data.get("language", "Urdu").strip()

    if not text:
        return jsonify({"error": "Nothing to translate."}), 400

    try:
        translated = translate_text(text, language)
        return jsonify({"translated": translated})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)