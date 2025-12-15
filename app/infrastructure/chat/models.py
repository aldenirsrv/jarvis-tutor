import os
import logging
from fastapi import HTTPException
from huggingface_hub import InferenceClient
from openai import OpenAI
from app.shared.prompts import get_prompt
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

    def run(self, user_input: str, language:str):
        # Detecção de tópico automática
        topic = detect_topic(user_input)
        logger.info("HuggingFaceChat | model=%s", self.model_name)

        # Monta contexto com base no tópico
        messages = self.build_messages(user_input, topic, language)

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

    def stream(self, user_input: str,  language:str):
        """
        Streaming de texto (se o provedor suportar stream=True).
        """
        try:
            topic = detect_topic(user_input)
            messages = self.build_messages(user_input, topic, language)
            logger.info("HuggingFaceChat.stream start | model=%s", self.model)
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=True,
                max_tokens=120,  # permitir respostas mais longas sem travar
            )
            for chunk in stream:
                delta = chunk.choices[0].delta
                piece = delta.get("content") if isinstance(delta, dict) else delta
                if piece:
                    yield piece
            logger.info("HuggingFaceChat.stream done | model=%s", self.model)
        except Exception as exc:
            logger.exception("HF chat stream failed: %s", exc)
            raise HTTPException(status_code=502, detail=str(exc))

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

    def build_messages(self, user_input: str, topic: str | None = None, language:str = 'en-US'):
        messages = []
        print(language)

        system_prompt = get_prompt(topic, language)
        messages.append({"role": "system", "content": system_prompt})

        if self.memory:
            history = self.memory.get_last_messages(limit=5)
            for role, content in history:
                messages.append({"role": role, "content": content})

        messages.append({"role": "user", "content": user_input})
        return messages

    def run(self, user_input: str, language:str):
        topic = detect_topic(user_input)
        messages = self.build_messages(user_input, topic, language)

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

    def stream(self, user_input: str, language):
        """
        Streaming de texto (se o provedor suportar stream=True).
        - Garante flush do primeiro token sem quebrar o iterador.
        - Lança 502 "Stream de texto vazio." se nada vier do provider.
        """
        import time
        try:
            topic = detect_topic(user_input)  # deve ser local/barato
            messages = self.build_messages(user_input, topic, language)

            logger.info("HuggingFaceChat.stream start | model=%s", self.model_name)
            t0 = time.perf_counter()
            got_any = False
            first_logged = False

            stream = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=400,          # keep it small
                temperature=0.7,
                stream=True,
            )

            for chunk in stream:
                # Compat com SDKs que retornam dict ou objeto com .content
                delta = getattr(chunk.choices[0].delta, "content", None)
                if delta is None:
                    d = chunk.choices[0].delta
                    delta = d.get("content") if isinstance(d, dict) else None
                if not delta:
                    continue

                if not first_logged:
                    first_logged = True
                    logger.info("first_token_ms=%.0f", (time.perf_counter() - t0) * 1000)

                got_any = True
                yield delta  # não quebra o loop; stream segue normalmente

            if not got_any:
                # Nenhum token útil veio do provider
                raise HTTPException(status_code=502, detail="Stream de texto vazio.")
        except HTTPException:
            # repassa exatamente como está
            raise
        except Exception as exc:
            logger.exception("HF chat stream failed: %s", exc)
            # mapeie erros de crédito se quiser
            msg = str(exc)
            if "402" in msg or "Payment Required" in msg or "credits" in msg.lower():
                raise HTTPException(status_code=402, detail=(
                    "You have exceeded your monthly included credits."
                )) from exc
            raise HTTPException(status_code=502, detail=str(exc)) from exc

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
