from typing import Any, Callable, TypedDict

RewardFunc = Callable[[list, list], list[float]]


class Document(TypedDict):
    id: str
    title: str
    body: str
    is_supporting: bool
    text: str


class Message(TypedDict):
    role: str
    content: str


class Input(TypedDict):
    prompt: list[Message]
    docs: list[Document]


class RunContext(TypedDict):
    input: Input
    trajectory: list[Message]


__all__ = [
    "Input",
    "Message",
    "RewardFunc",
    "RunContext",
]
