import html
import re


TOPICS_KEYWORDS = {
    "job_interview": ["interview", "job interview", "practice interview"],
    "travel": ["travel", "trip", "airport", "hotel"],
    "vocabulary": ["vocabulary", "words", "idioms", "learn new words"],
    "daily_conversation": ["chat", "talk", "conversation", "daily talk"],
}

def detect_topic(user_message):
    message = user_message.lower()
    for topic, keywords in TOPICS_KEYWORDS.items():
        if any(keyword in message for keyword in keywords):
            return topic
    return None


def clean_lesson_text(raw_text: str) -> str:
    if not raw_text:
        return ""

    text = html.unescape(raw_text)

    # Line breaks and list items first to preserve structure.
    text = re.sub(r"<\s*br\s*/?\s*>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<\s*li[^>]*>", "\n- ", text, flags=re.IGNORECASE)
    text = re.sub(r"</\s*li\s*>", "\n", text, flags=re.IGNORECASE)

    # Headings and paragraphs as block separators.
    text = re.sub(r"<\s*h[1-6][^>]*>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"</\s*h[1-6]\s*>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<\s*p[^>]*>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"</\s*p\s*>", "\n", text, flags=re.IGNORECASE)

    # Generic block tags.
    text = re.sub(r"<\s*(div|ul|ol)[^>]*>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"</\s*(div|ul|ol)\s*>", "\n", text, flags=re.IGNORECASE)

    # Drop any remaining tags.
    text = re.sub(r"<[^>]+>", "", text)

    # Normalize whitespace.
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
