import re

def add_natural_pauses(text: str) -> str:
    # 1. Quebra o texto em sentenças básicas
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())

    processed = []
    for sentence in sentences:
        sentence = sentence.strip()

        # 2. Se a frase for muito curta, adiciona "..." no final
        if len(sentence.split()) <= 5:
            sentence += "..."

        # 3. Se for média (6–10 palavras), quebra com vírgula ou "..."
        elif 6 <= len(sentence.split()) <= 10 and not sentence.endswith("..."):
            sentence += "..."

        # 4. Frases maiores mantêm a pontuação normal
        processed.append(sentence)

    # 5. Junta com espaçamento natural
    return " ".join(processed)