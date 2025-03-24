import spacy
import json

nlp = spacy.load("en_core_web_sm")

input_file = "EAE/response.json"
output_file = "EAE/response.json"

with open(input_file, "r", encoding="utf-8") as file:
    data = json.load(file)


def extract_core_words(text):
    if not isinstance(text, str):
        text = str(text) 

    if len(text.split()) == 1:
        return None

    doc = nlp(text)
    core_words = [token.text for token in doc if token.pos_ in ['NOUN', 'PROPN']]
    return core_words if core_words else None


def extract_head_words(text):
    if not text or not isinstance(text, str) or text.strip() == "":
        return None

    if len(text.split()) == 1:
        return None

    try:
        doc = nlp(text)
        head_words = list(set([token.head.text for token in doc if token != token.head]))
        return head_words if head_words else None
    except Exception as e:
        print(f"Error extracting head words: {e}")
        return None


for entry in data:
    for event in entry.get("arguments", []):
        original_text = event.get("text")
        if original_text:
            core_words = extract_core_words(original_text)
            if core_words:
                event["core_words"] = core_words

            head_words = extract_head_words(original_text)
            if head_words:
                event["head_words"] = head_words

            if not core_words and "core_words" in event:
                del event["core_words"]
            if not head_words and "head_words" in event:
                del event["head_words"]

with open(output_file, "w", encoding="utf-8") as file:
    json.dump(data, file, indent=4, ensure_ascii=False)

print("数据处理完成，已保存到:", output_file)
