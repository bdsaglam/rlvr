"""MuSiQue evaluation metrics and utility functions."""

import re
import string
from collections import Counter
from typing import List


def normalize_answer(s: str) -> str:
    """Normalize answer for comparison."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def exact_match(prediction: str, references: List[str]) -> float:
    """Compute exact match score."""
    normalized_prediction = normalize_answer(prediction)
    for reference in references:
        if normalized_prediction == normalize_answer(reference):
            return 1.0
    return 0.0


def f1(prediction: str, references: List[str]) -> float:
    """Compute F1 score against references."""
    normalized_prediction = normalize_answer(prediction)
    max_f1 = 0.0
    
    for reference in references:
        normalized_reference = normalize_answer(reference)
        
        pred_tokens = normalized_prediction.split()
        ref_tokens = normalized_reference.split()
        
        if len(pred_tokens) == 0 or len(ref_tokens) == 0:
            if len(pred_tokens) == len(ref_tokens):
                f1_score = 1.0
            else:
                f1_score = 0.0
        else:
            common = Counter(pred_tokens) & Counter(ref_tokens)
            num_same = sum(common.values())
            
            precision = 1.0 * num_same / len(pred_tokens)
            recall = 1.0 * num_same / len(ref_tokens)
            
            if precision + recall == 0:
                f1_score = 0.0
            else:
                f1_score = (2 * precision * recall) / (precision + recall)
        
        max_f1 = max(max_f1, f1_score)
    
    return max_f1


def get_last_answer(completion) -> str:
    """Extract the last answer from completion messages."""
    if isinstance(completion, list):
        # Find the last assistant message
        for message in reversed(completion):
            if message.get("role") == "assistant":
                content = message.get("content", "")
                # Extract answer from <answer> tags
                answer_match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL | re.IGNORECASE)
                if answer_match:
                    return answer_match.group(1).strip()
    elif isinstance(completion, str):
        # Extract answer from string
        answer_match = re.search(r'<answer>(.*?)</answer>', completion, re.DOTALL | re.IGNORECASE)
        if answer_match:
            return answer_match.group(1).strip()
    
    return ""


def extract_retrieved_doc_ids(content: str) -> List[str]:
    """Extract document IDs from tool response content."""
    return [id.strip() for id in re.findall(r"^Document ID: (\S+)", content, re.MULTILINE)]


def extract_all_retrieved_doc_ids(completion):
    """Extract all retrieved document IDs from a completion."""
    retrieved_ids = set()
    
    if isinstance(completion, list):
        for message in completion:
            if message.get("role") == "tool":
                content = message.get("content", "")
                ids = extract_retrieved_doc_ids(content)
                retrieved_ids.update(ids)
    elif isinstance(completion, str):
        ids = extract_retrieved_doc_ids(completion)
        retrieved_ids.update(ids)
    
    return list(retrieved_ids)