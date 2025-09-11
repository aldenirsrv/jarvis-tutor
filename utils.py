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