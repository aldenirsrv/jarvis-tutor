import edge_tts
import asyncio
import base64

class EdgeTTS:
    def __init__(self, voice="en-US-GuyNeural"):
        self.voice = voice

    async def run_async(self, text: str) -> bytes:
        communicate = edge_tts.Communicate(text, voice=self.voice)
        audio_chunks = []
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_data = chunk.get("data")
                if audio_data:
                    audio_chunks.append(audio_data)
        return b"".join(audio_chunks)

    def run(self, text: str) -> bytes:
        return asyncio.run(self.run_async(text))