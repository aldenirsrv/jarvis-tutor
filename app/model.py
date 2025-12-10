import os
import logging
from fastapi import HTTPException
from huggingface_hub import InferenceClient
from openai import OpenAI
from app.prompts import get_prompt
from utils import detect_topic  # import da sua função de detecção

logger = logging.getLogger(__name__)
from dotenv import load_dotenv
load_dotenv()

class HuggingFaceChat:
    def __init__(self, memory=None, model_name="mistralai/Mistral-7B-Instruct-v0.3", hf_token=None):
        self.memory = memory
        # Permite override via variável de ambiente
        self.model_name = os.getenv("HUGGINGFACE_MODEL", model_name)
        self.hf_token = hf_token
        self.provider = os.getenv("HUGGINGFACE_PROVIDER")  # opcional (ex: novita, hf-inference)
        self.client = self._create_client(self.provider)
        self.model = self.model_name

    def _create_client(self, provider: str | None):
        if provider:
            return InferenceClient(api_key=self.hf_token, provider=provider)
        return InferenceClient(api_key=self.hf_token)

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

        tried_providers = []
        # ordem: provider explicitado -> hf-inference -> auto (None)
        candidate_providers = [
            self.provider,
            "hf-inference",
            None,
        ]

        completion = None
        response_text = None
        last_exc = None

        for provider in candidate_providers:
            if provider in tried_providers:
                continue
            tried_providers.append(provider)

            try:
                client = self._create_client(provider)
                response_text = None
                completion = client.chat.completions.create(
                    model=self.model,
                    messages=messages
                )
                response_text = completion.choices[0].message["content"].strip()
            except Exception as exc:
                # Alguns modelos não suportam 'conversational' — tenta text_generation
                msg = str(exc)
                if "task 'conversational'" in msg:
                    prompt = self._messages_to_prompt(messages)
                    try:
                        tg = client.text_generation(
                            prompt,
                            model=self.model,
                            max_new_tokens=512,
                        )
                        if isinstance(tg, str):
                            response_text = tg.strip()
                        elif isinstance(tg, dict):
                            response_text = tg.get("generated_text", "").strip()
                        else:
                            response_text = str(tg).strip()
                    except Exception as exc2:
                        last_exc = exc2
                        continue
                else:
                    last_exc = exc
                    continue

            if response_text:
                self.client = client  # guarda o cliente que funcionou
                self.provider = provider
                break

        if response_text is None:
            msg = str(last_exc) if last_exc else "Falha ao chamar Inference API"
            if "task 'text-generation'" in msg or "task 'conversational'" in msg:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Model '{self.model}' não está habilitado para chat/text-generation "
                        "na Inference API. Defina HUGGINGFACE_MODEL para um modelo compatível "
                        "ou configure HUGGINGFACE_PROVIDER."
                    ),
                )
            if "Unauthorized" in msg or "api_key" in msg or "Payment Required" in msg:
                raise HTTPException(
                    status_code=402,
                    detail="Token HF não autorizado para inferência. Defina HUGGINGFACEHUB_API_TOKEN com acesso.",
                )
            raise HTTPException(status_code=502, detail=msg)

        response = response_text

        # Armazena no banco
        if self.memory:
            self.memory.add_message("user", user_input)
            self.memory.add_message("assistant", response)

        return response

    def _messages_to_prompt(self, messages) -> str:
        """
        Converte mensagens em um prompt de texto simples para text_generation.
        """
        lines = []
        for m in messages:
            role = m.get("role", "user")
            prefix = {
                "system": "System",
                "assistant": "Assistant",
                "user": "User",
            }.get(role, role.capitalize())
            lines.append(f"{prefix}: {m.get('content', '')}")
        lines.append("Assistant:")
        return "\n".join(lines)


class OpenAIChat:
    """
    Alternativa usando OpenAI Chat Completions (gpt-4o-mini, etc).
    """

    def __init__(self, memory=None, model_name: str = "gpt-4o-mini", api_key: str | None = None):
        self.memory = memory
        self.model_name = os.getenv("OPENAI_MODEL", model_name)
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise HTTPException(
                status_code=401,
                detail="OPENAI_API_KEY não definido no ambiente.",
            )
        self.client = OpenAI(api_key=self.api_key)

    def build_messages(self, user_input: str, topic: str | None = None):
        messages = []

        system_prompt = get_prompt(topic)
        messages.append({"role": "system", "content": system_prompt})

        if self.memory:
            history = self.memory.get_last_messages(limit=5)
            for role, content in history:
                messages.append({"role": role, "content": content})

        messages.append({"role": "user", "content": user_input})
        return messages

    def run(self, user_input: str):
        topic = detect_topic(user_input)
        messages = self.build_messages(user_input, topic)

        try:
            logger.info("OpenAIChat.run start | model=%s", self.model_name)
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
            )
            response = completion.choices[0].message.content.strip()
            logger.info("OpenAIChat.run done | model=%s | chars=%d", self.model_name, len(response))
        except Exception as exc:
            raise HTTPException(status_code=502, detail=str(exc))

        if self.memory:
            self.memory.add_message("user", user_input)
            self.memory.add_message("assistant", response)

        return response

    def stream(self, user_input: str):
        """
        Gera a resposta em streaming (chunks de texto).
        """
        topic = detect_topic(user_input)
        messages = self.build_messages(user_input, topic)
        buffer: list[str] = []

        try:
            logger.info("OpenAIChat.stream start | model=%s", self.model_name)
            stream = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if not delta:
                    continue
                if isinstance(delta, list):
                    piece = "".join(delta)
                else:
                    piece = delta
                buffer.append(piece)
                yield piece
            logger.info("OpenAIChat.stream done | model=%s | chars=%d", self.model_name, len("".join(buffer)))
        except Exception as exc:
            raise HTTPException(status_code=502, detail=str(exc))
        finally:
            if buffer:
                response = "".join(buffer).strip()
                if self.memory:
                    self.memory.add_message("user", user_input)
                    self.memory.add_message("assistant", response)


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
