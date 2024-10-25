import os

from fastapi import FastAPI
from pydantic import BaseModel

from bot import LangChainAgent

app = FastAPI()


class Message(BaseModel):
    content: str


@app.post("/stream_chat/")
async def stream_chat(message: Message):
    answer = LangChainAgent(
        os.environ["DB_URL"],
        os.environ["OPENAI_API_KEY"],
        ).query(message.content)

    return Message(content=answer["agent"]["messages"][0].content)
