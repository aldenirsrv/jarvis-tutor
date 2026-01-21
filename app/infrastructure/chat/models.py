import os
import logging
from fastapi import HTTPException
from huggingface_hub import InferenceClient
from openai import OpenAI
from app.shared.prompts import get_prompt
from utils import detect_topic, clean_lesson_text  # import da sua função de detecção
from app.infrastructure.memory.sqlite_lessons_impl import SQLiteLessons
logger = logging.getLogger(__name__)
from dotenv import load_dotenv
load_dotenv()

lessons = SQLiteLessons()
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
    
    # def build_messages(self, user_input: str, topic: str | None = None, language: str = "en-US"):
    #     messages: list[dict[str, str]] = []

    #     system_prompt = get_prompt(topic, language, history)
    #     messages.append({"role": "system", "content": system_prompt})

    #     if self.memory:
    #         history = self.memory.get_last_messages(limit=5)  # expected: list[tuple[str, str]]
    #         for role, content in history:
    #             if not content:
    #                 continue

    #             # Only allow roles that the Chat Completions API expects
    #             if role not in ("user", "assistant", "system"):
    #                 continue

    #             # Ensure it's a string and not huge
    #             messages.append({"role": role, "content": str(content)[:4000]})

    #     messages.append({"role": "user", "content": str(user_input)})
    #     return messages
    def build_messages(self, user_input: str, topic: str | None = None, language: str = "en-US", lesson:dict = None):
        if self.memory:
            history = self.memory.get_last_messages(limit=30)  # [(role, content), ...]
            lines = []
            for role, content in history:
                if role not in ("user", "assistant"):
                    continue
                if not content:
                    continue
                lines.append(f"{role.upper()}: {str(content).strip()[:10000]}")
        system_prompt = get_prompt(topic, language, history=history, lesson= lesson)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": str(user_input)},
        ]
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

    def stream(self, user_input: str, language, selected_lesson:str = None):
        """
        Streams text tokens from the provider (stream=True).
        - Logs time to first token
        - Raises 502 if no usable text tokens arrive
        """
        import time

        try:
            topic = detect_topic(user_input)  # should be local/cheap
            if selected_lesson:
                lesson = lessons.get_lesson(lesson_id=selected_lesson)
                #  content = lesson.get("content", "")
                # goals = lesson.get("goals", "")
                # rules = lesson.get("rules", "")
            messages = self.build_messages(user_input, topic, language, lesson)
            answer_parts: list[str] = []

            logger.info("OpenAIChat.stream start | model=%s", self.model_name)
            t0 = time.perf_counter()
            got_any = False
            first_logged = False

            stream = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=400,
                temperature=0.7,
                stream=True,
            )

            for chunk in stream:
                choice = chunk.choices[0]
                d = choice.delta  # can contain role/tool_calls/etc.

                # Prefer attribute access; fall back to dict-like if needed
                delta_text = getattr(d, "content", None)
                if delta_text is None and isinstance(d, dict):
                    delta_text = d.get("content")

                # Normalize None/"" to "no token"
                if not delta_text:
                    continue

                if not first_logged:
                    first_logged = True
                    logger.info("first_token_ms=%.0f", (time.perf_counter() - t0) * 1000)

                got_any = True
                answer_parts.append(delta_text)
                yield delta_text

            answer = "".join(answer_parts)
            if self.memory:
                self.memory.add_message("user", user_input)
                self.memory.add_message("assistant", answer)
            # print(f"### __ {user_input} __ ### Answer {answer}")

            if not got_any:
                raise HTTPException(status_code=502, detail="Text stream empty.")

        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("Chat stream failed: %s", exc)
            msg = str(exc)
            if "402" in msg or "Payment Required" in msg or "credits" in msg.lower():
                raise HTTPException(
                    status_code=402,
                    detail="You have exceeded your monthly included credits.",
                ) from exc
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
