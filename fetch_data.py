import requests
from bs4 import BeautifulSoup
import json
import time
import os

topics = [
    "Migraine", "Headache", "Asthma", "High Blood Pressure", "Common Cold",
    "Anxiety", "Depression", "Diabetes", "Allergies", "Flu",
    "Back Pain", "Eczema", "Acid Reflux", "Insomnia", "Urinary Tract Infection",
    "Anemia", "Dizziness and Vertigo", "Chronic Pain", "Eye Diseases", "Caffeine",
    "Arthritis", "Osteoporosis", "Obesity", "High Cholesterol", "Heart Disease",
    "Stroke", "Kidney Disease", "Liver Disease", "Hepatitis", "Thyroid Disease",
    "Psoriasis", "Acne", "Sinusitis", "Bronchitis", "Pneumonia",
    "Tuberculosis", "COPD", "Sleep Apnea", "Constipation", "Diarrhea",
    "Irritable Bowel Syndrome", "Food Poisoning", "Lactose Intolerance", "Celiac Disease", "Gallstones",
    "Hemorrhoids", "Varicose Veins", "Anemia", "Vitamin D Deficiency", "Iron Deficiency",
    "Menstrual Cramps", "Menopause", "Pregnancy", "Morning Sickness", "Erectile Dysfunction",
    "Prostate Problems", "Hair Loss", "Dandruff", "Cold Sores", "Fungal Infections",
    "Sunburn", "Bee Stings", "Food Allergies", "Lactose Intolerance", "Motion Sickness"
]
def clean_html(raw_html):
    soup = BeautifulSoup(raw_html, "lxml")
    return soup.get_text(separator=" ", strip=True)

def fetch_topic(topic_name, retries=3):
    url = "https://wsearch.nlm.nih.gov/ws/query"
    params = {"db": "healthTopics", "term": topic_name}
    for attempt in range(retries):
        try:
            response = requests.get(url, params=params, timeout=10)
            return response.text
        except requests.exceptions.ConnectionError:
            print(f"  Connection error, retrying ({attempt + 1}/{retries})...")
            time.sleep(2)
    print(f"  Failed after {retries} attempts: {topic_name}")
    return None

def parse_topic(xml_text, topic_name):
    if not xml_text:
        return None
    soup = BeautifulSoup(xml_text, "xml")
    document = soup.find("document")

    if not document:
        print(f"  No result found for: {topic_name}")
        return None

    title_tag = document.find("content", {"name": "title"})
    summary_tag = document.find("content", {"name": "FullSummary"})
    url = document.get("url")

    if not title_tag or not summary_tag:
        print(f"  Missing data for: {topic_name}")
        return None

    return {
        "topic": clean_html(str(title_tag)),
        "summary": clean_html(str(summary_tag)),
        "source": "MedlinePlus",
        "url": url
    }

os.makedirs("data", exist_ok=True)
output_path = "data/health_topics.json"

if os.path.exists(output_path):
    with open(output_path, "r", encoding="utf-8") as f:
        all_data = json.load(f)
    already_fetched = {item["topic"].lower() for item in all_data}
else:
    all_data = []
    already_fetched = set()

for topic in topics:
    if topic.lower() in already_fetched:
        print(f"Skipping (already fetched): {topic}")
        continue

    print(f"Fetching: {topic}...")
    xml_data = fetch_topic(topic)
    result = parse_topic(xml_data, topic)
    if result:
        all_data.append(result)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(all_data, f, indent=2, ensure_ascii=False)

    time.sleep(1.5)

print(f"\nDone! Saved {len(all_data)} topics to {output_path}")