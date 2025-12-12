import re

def add_natural_pauses(text: str) -> str:
    # 1. Quebra o texto em sentenças básicas
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())

    processed = []
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        words = sentence.split()
        n = len(words)

        # 2. Se a frase for muito curta, adiciona "..." no final
        if n <= 5:
            sentence += "..."

        # 3. Se for média (6–10 palavras), acrescenta "..."
        elif 6 <= n <= 10 and not sentence.endswith("..."):
            sentence += "..."

        # 4. Frases maiores: insere uma vírgula como pausa se não houver
        elif n >= 11 and "," not in sentence:
            mid = n // 2
            sentence = " ".join(words[:mid]) + ", " + " ".join(words[mid:])

        processed.append(sentence)

    # 5. Junta com espaçamento natural
    return " ".join(processed)
