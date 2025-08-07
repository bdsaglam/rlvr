from typing import List

from verifiers.models import Message
from verifiers.parsers.xml_parser import XMLParser


def get_assistant_messages(trajectory: List[Message]) -> List[Message]:
    """Helper function to extract assistant messages from a trajectory."""
    return [msg for msg in trajectory if msg["role"] == "assistant"]


def get_last_assistant_message(trajectory: List[Message]) -> str | None:
    """Extract the last assistant message from a trajectory."""
    for msg in reversed(trajectory):
        if msg["role"] == "assistant":
            return msg["content"]
    return None


def get_last_answer(trajectory: List[Message], tag: str = "answer") -> str | None:
    """Extract the last answer from a trajectory."""
    parser = XMLParser(fields=[tag])
    for msg in reversed(trajectory):
        if msg["role"] == "assistant":
            parsed = parser.parse(msg["content"])
            if answer := getattr(parsed, tag, None):
                return answer
    return None
