from typing import List

import nltk
from nltk.corpus import words
from nltk.tokenize import word_tokenize

from verifiers.models import Message
from verifiers.parsers.xml_parser import XMLParser

# First-time setup
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("words")

# Load English dictionary
EN_VOCAB = set(w.lower() for w in words.words())


def rate_text_for_language_use(text: str) -> float:
    # Tokenize the text
    tokens = word_tokenize(text)

    # Keep only alphabetic tokens (strip punctuation, numbers, etc.)
    word_tokens = [t.lower() for t in tokens if t.isalpha()]

    if not word_tokens:
        return 0.0  # No usable words

    # Count how many are in the English dictionary
    english_count = sum(1 for word in word_tokens if word in EN_VOCAB)

    # Return normalized reward
    return english_count / len(word_tokens)


def natural_language_reward_func(
    completions: List[List[Message]],
    **kwargs,
) -> List[float]:
    """Reward function that checks if the language of the completion follows grammar rules and is understandable."""
    tag = "think"
    parser = XMLParser(fields=[tag])
    rewards: List[float] = []
    for trajectory in completions:
        assistant_messages = [msg for msg in trajectory if msg["role"] == "assistant"]
        thinking_traces = [getattr(parser.parse(msg["content"]), tag, None) for msg in assistant_messages]
        thinking = "\n".join([t for t in thinking_traces if t is not None])
        reward = rate_text_for_language_use(thinking) * 0.2
        rewards.append(reward)
    return rewards
