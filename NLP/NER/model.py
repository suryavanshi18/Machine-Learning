import re
import emoji
import torch
from gliner import GLiNER
from deep_translator import GoogleTranslator

LABELS        = ["PERSON", "ORG", "PRODUCT", "EVENT", "LOC"]
WINDOW_SIZE   = 1000
OVERLAP       = 200
NER_THRESHOLD = 0.5
MAX_TEXT_LEN  = 4500
TRANSLATE_LIMIT = 4500

class NERPipeline:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading GLiNER on {self.device}...")
        self.model = GLiNER.from_pretrained("urchade/gliner_small-v2.1")
        self.model = self.model.to(self.device)
        self.model.eval()
        print("GLiNER loaded.")

    # --------------------------------------------------------
    # PREPROCESSING
    # --------------------------------------------------------
    def extract_metadata(self, text):
        mentions = re.findall(r'@(\w+)', text)
        hashtags = re.findall(r'#(\w+)', text)
        urls     = re.findall(r'https?://\S+', text)
        quotes   = re.findall(r'["""](.+?)["""]', text)
        return mentions, hashtags, urls, quotes

    def clean_text(self, text):
        text = re.sub(r'https?://\S+', '', text)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#\w+', '', text)
        text = emoji.replace_emoji(text, '')
        text = re.sub(r'~\s*copy\b.*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def normalize_caps(self, text):
        def cap_word(m):
            word = m.group(0)
            return word.title() if word.isupper() and len(word) > 1 else word
        return re.sub(r'\b\w+\b', cap_word, text)

    def translate_to_english(self, text):
        if not text.strip():
            return ""
        if len(text) <= TRANSLATE_LIMIT:
            try:
                return GoogleTranslator(source='auto', target='en').translate(text)
            except:
                return text
        chunks = [text[i:i+TRANSLATE_LIMIT] for i in range(0, len(text), TRANSLATE_LIMIT)]
        translated = []
        for chunk in chunks:
            try:
                translated.append(
                    GoogleTranslator(source='auto', target='en').translate(chunk)
                )
            except:
                translated.append(chunk)
        return " ".join(translated)

    def parse_hashtags(self, hashtags):
        return [re.sub(r'([A-Z][a-z]+|\d+)', r' \1', tag).strip() for tag in hashtags]

    # --------------------------------------------------------
    # NER
    # --------------------------------------------------------
    def sliding_window_entities(self, text):
        text = text[:MAX_TEXT_LEN]
        if len(text) <= WINDOW_SIZE:
            return self.model.predict_entities(text, LABELS, threshold=NER_THRESHOLD)
        all_entities = []
        start = 0
        while start < len(text):
            end   = min(start + WINDOW_SIZE, MAX_TEXT_LEN)
            chunk = text[start:end]
            if not chunk.strip():
                break
            for e in self.model.predict_entities(chunk, LABELS, threshold=NER_THRESHOLD):
                all_entities.append({
                    "text":  e["text"].strip(),
                    "label": e["label"],
                    "score": round(e["score"], 3),
                    "start": e["start"] + start,
                    "end":   e["end"]   + start,
                })
            start += (WINDOW_SIZE - OVERLAP)
            if end >= MAX_TEXT_LEN:
                break
        return all_entities

    def deduplicate_entities(self, entities):
        seen = {}
        for e in entities:
            key = (e["text"].lower(), e["label"])
            if key not in seen or e["score"] > seen[key]["score"]:
                seen[key] = e
        return [{"text": v["text"], "label": v["label"], "score": v["score"]}
                for v in seen.values()]

    def group_by_label(self, entities):
        grouped = {
            "ner_persons": [], "ner_orgs": [], "ner_locations": [],
            "ner_events": [], "ner_products": []
        }
        label_map = {
            "PERSON": "ner_persons", "ORG": "ner_orgs", "LOC": "ner_locations",
            "EVENT": "ner_events",   "PRODUCT": "ner_products"
        }
        for e in entities:
            key = label_map.get(e["label"])
            if key:
                grouped[key].append(e["text"])
        return grouped

    # --------------------------------------------------------
    # PREDICT
    # --------------------------------------------------------
    def predict(self, title: str, text: str) -> dict:
        raw_text = f"{str(title)}. {str(text)}"

        mentions, hashtags, urls, quotes = self.extract_metadata(raw_text)
        cleaned      = self.clean_text(raw_text)
        normalized   = self.normalize_caps(cleaned)
        translated   = self.translate_to_english(normalized)
        hashtag_text = " ".join(self.parse_hashtags(hashtags))
        quote_text   = " ".join(quotes)
        full_text    = f"{translated}. {hashtag_text}. {quote_text}".strip()[:MAX_TEXT_LEN]

        raw_entities = self.sliding_window_entities(full_text)
        entities     = self.deduplicate_entities(raw_entities)

        ner_texts = {e["text"].lower() for e in entities}
        for m in mentions:
            name = re.sub(r'[_0-9]', ' ', m).strip()
            if name and name.lower() not in ner_texts:
                entities.append({"text": name, "label": "PERSON", "score": 0.75})
        for q in quotes:
            if q.lower() not in ner_texts:
                entities.append({"text": q, "label": "PRODUCT", "score": 0.70})

        grouped = self.group_by_label(entities)

        return {
            "original_text":   raw_text,
            "translated_text": translated,
            "mentions":        mentions,
            "hashtags":        hashtags,
            "entities":        entities,
            **grouped
        }

    def predict_batch(self, items: list) -> list:
        return [self.predict(i.get("title", ""), i.get("text", "")) for i in items]