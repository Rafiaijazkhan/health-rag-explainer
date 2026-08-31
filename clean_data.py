import json
import re

with open("data/health_topics.json", "r", encoding="utf-8") as f:
    data = json.load(f)

def strip_tags(text):
    """Remove any leftover HTML tags using regex, then clean up spacing."""
    text = re.sub(r"<[^>]+>", "", text)  # remove anything like <span...> or </span>
    text = re.sub(r"\s+", " ", text)      # collapse extra spaces
    return text.strip()

cleaned = {}  # use dict to automatically deduplicate by cleaned topic name

for item in data:
    clean_topic = strip_tags(item["topic"])
    clean_summary = strip_tags(item["summary"])
    key = clean_topic.lower()

    cleaned[key] = {
        "topic": clean_topic,
        "summary": clean_summary,
        "source": item["source"],
        "url": item["url"]
    }

final_data = list(cleaned.values())

with open("data/health_topics.json", "w", encoding="utf-8") as f:
    json.dump(final_data, f, indent=2, ensure_ascii=False)

print(f"Cleaned! {len(data)} entries -> {len(final_data)} unique, clean entries.")
for item in final_data:
    print("-", item["topic"])