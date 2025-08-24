"""Data loading and processing utilities for evaluation results"""

import json
import pandas as pd
import re
from typing import List, Dict, Any

def load_results(file_path: str) -> List[Dict[str, Any]]:
    """Load evaluation results from a JSONL file"""
    results = []
    with open(file_path, "r") as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line.strip()))
    return results

def process_data(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """Convert results to a pandas DataFrame with derived columns"""
    df = pd.DataFrame(results)
    
    # Extract key information
    df["question"] = df["prompt"].apply(extract_question)
    df["predicted_answer"] = df['answer']
    df["reference_answers"] = df["info"].map(lambda x: str(x["answers"]))
    df["n_hops"] = df["info"].map(lambda x: x["n_hops"])
    
    # Conversation analysis
    df["n_turns"] = df["completion"].apply(count_turns)
    df["n_tool_calls"] = df["completion"].apply(count_tool_calls)
    df["used_supporting_docs"] = df.apply(
        lambda row: check_supporting_docs_usage(row["completion"], row["info"]["docs"]), axis=1
    )
    
    # Ensure all reward columns exist (for backward compatibility)
    reward_columns = [
        'reward', 'exact_match_reward', 'f1_reward', 'retrieval_recall_reward', 
        'citation_reward', 'format_reward', 'combined_reward'
    ]
    for col in reward_columns:
        if col not in df.columns:
            df[col] = 0.0
    
    # For legacy compatibility, create exact_match from exact_match_reward if it doesn't exist
    if 'exact_match' not in df.columns:
        df['exact_match'] = (df['exact_match_reward'] > 0).astype(int)
    
    return df

def extract_question(prompt: List[Dict[str, str]]) -> str:
    """Extract the question from the prompt"""
    if len(prompt) > 1 and "content" in prompt[1]:
        content = prompt[1]["content"]
        # Get the first line which should be the question
        return content.split("\n")[0]
    return ""

def count_turns(completion: List[Dict[str, Any]]) -> int:
    """Count conversation turns (assistant messages)"""
    return len([msg for msg in completion if msg.get("role") == "assistant"])

def count_tool_calls(completion: List[Dict[str, Any]]) -> int:
    """Count total tool calls made"""
    count = 0
    for msg in completion:
        if msg.get("tool_calls"):
            count += len(msg["tool_calls"])
    return count

def check_supporting_docs_usage(completion: List[Dict[str, Any]], docs: List[Dict[str, Any]]) -> float:
    """Check if supporting documents were retrieved and used"""
    supporting_doc_ids = [str(doc["id"]) for doc in docs if doc.get("is_supporting", False)]
    
    # Look for document IDs mentioned in tool responses
    used_doc_ids = set()
    for msg in completion:
        if msg.get("role") == "tool" and "content" in msg:
            content = msg["content"]
            # Extract document IDs from tool responses
            doc_id_matches = re.findall(r"Document ID: (\d+)", content)
            used_doc_ids.update(doc_id_matches)
    
    # Check how many supporting docs were used
    used_supporting = len(set(supporting_doc_ids) & used_doc_ids)
    total_supporting = len(supporting_doc_ids)
    
    return used_supporting / total_supporting if total_supporting > 0 else 0

def get_error_patterns(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze common error patterns"""
    incorrect = df[df['exact_match'] == 0]
    
    patterns = {
        'total_errors': len(incorrect),
        'error_rate_by_hops': df.groupby('n_hops')['exact_match'].apply(lambda x: 1 - x.mean()).to_dict(),
        'avg_tools_correct': df[df['exact_match'] == 1]['n_tool_calls'].mean(),
        'avg_tools_incorrect': df[df['exact_match'] == 0]['n_tool_calls'].mean(),
        'support_docs_correct': df[df['exact_match'] == 1]['used_supporting_docs'].mean(),
        'support_docs_incorrect': df[df['exact_match'] == 0]['used_supporting_docs'].mean(),
        'no_answer_count': len(df[df['predicted_answer'] == '']),
    }
    
    # Reward component analysis
    reward_cols = ['exact_match_reward', 'f1_reward', 'retrieval_recall_reward', 
                   'citation_reward', 'format_reward', 'combined_reward']
    
    reward_comparison = {}
    for col in reward_cols:
        if col in df.columns:
            correct_mean = df[df['exact_match'] == 1][col].mean()
            incorrect_mean = df[df['exact_match'] == 0][col].mean()
            reward_comparison[col] = {
                'correct': correct_mean,
                'incorrect': incorrect_mean,
                'difference': correct_mean - incorrect_mean
            }
    
    patterns['reward_comparison'] = reward_comparison
    
    return patterns