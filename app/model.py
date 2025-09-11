from huggingface_hub import InferenceClient
from app.prompts import get_prompt
from utils import detect_topic  # import da sua função de detecção

class HuggingFaceChat:
    def __init__(self, memory=None, model_name="mistralai/Mistral-7B-Instruct-v0.3", hf_token=None):
        self.memory = memory
        self.model_name = model_name
        self.hf_token = hf_token
        self.client = InferenceClient(api_key=self.hf_token, provider="novita")
        self.model = self.model_name

    def build_messages(self, user_input: str, topic: str | None = None):
        messages = []

        # Prompt temático (detectado)
        system_prompt = get_prompt(topic)
        messages.append({"role": "system", "content": system_prompt})

        # Histórico da conversa
        if self.memory:
            history = self.memory.get_last_messages(limit=5)
            for role, content in history:
                messages.append({"role": role, "content": content})

        # Entrada atual
        messages.append({"role": "user", "content": user_input})

        return messages

    def run(self, user_input: str):
        # Detecção de tópico automática
        topic = detect_topic(user_input)

        # Monta contexto com base no tópico
        messages = self.build_messages(user_input, topic)

        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages
        )

        response = completion.choices[0].message["content"].strip()

        # Armazena no banco
        if self.memory:
            self.memory.add_message("user", user_input)
            self.memory.add_message("assistant", response)

        return response


# class HuggingFaceChat:
#     def __init__(self, memory, model_name="mistralai/Mistral-7B-Instruct-v0.1", hf_token=None):
#         self.memory = memory
#         self.model_name = model_name
#         self.hf_token = hf_token
#         self.llm = HuggingFaceHub(
#             repo_id=model_name,
#             huggingfacehub_api_token=hf_token,
#             model_kwargs={"temperature": 0.7, "max_length": 512},
#         )

#     def run(self, prompt: str):
#         return self.llm(prompt)