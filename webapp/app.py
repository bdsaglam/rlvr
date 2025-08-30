#!/usr/bin/env -S uv run --script
# /// script
# dependencies = [
#   "python-fasthtml",
#   "pandas", 
#   "uvicorn",
#   "python-multipart"
# ]
# ///

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
from fasthtml.common import *
from data_loader import load_results, process_data
from components import (
    dashboard_card, results_table, inspect_view, 
    metric_breakdown, error_analysis_section, simple_records_list
)

# Initialize FastHTML app with Tailwind CSS
app, rt = fast_app(
    pico=False,
    hdrs=[
        Link(rel="stylesheet", href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css"),
        Link(rel="stylesheet", href="/static/style.css"),
        Script(src="https://unpkg.com/htmx.org@1.9.6"),
        Script(src="https://cdn.jsdelivr.net/npm/chart.js"),
        Script(src="/static/app.js")
    ]
)

# Global data storage - in production you'd use a proper database
data_store = {}

def load_evaluation_data(file_path: str):
    """Load and process evaluation results"""
    try:
        results = load_results(file_path)
        df = process_data(results)
        return df, results
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None

@rt("/")
def index():
    """Main dashboard page"""
    # Check if data is already loaded
    if 'df' not in data_store:
        # Try to load results.jsonl from the webapp folder
        results_path = Path(__file__).parent / "results.jsonl"
        if results_path.exists():
            df, raw_results = load_evaluation_data(str(results_path))
            if df is not None:
                data_store['df'] = df
                data_store['raw_results'] = raw_results
    
    if 'df' in data_store:
        df = data_store['df']
        dashboard_content = get_dashboard_content(df)
    else:
        dashboard_content = Div("No results.jsonl file found in webapp folder", cls="text-red-600 text-center p-8")
    
    return Div(
            # Header
            Div(
                Div(
                    H1("Multi-Hop QA Inspector", cls="text-2xl font-bold text-gray-900"),
                    P("Analyze model performance and inspect reasoning trajectories", cls="text-gray-600 text-sm"),
                    cls="flex-1"
                ),
                cls="flex items-center justify-between pb-4 mb-6 border-b border-gray-200"
            ),
            
            # Dashboard content
            dashboard_content,
            
            cls="max-w-7xl mx-auto p-6"
        )

def get_dashboard_content(df: pd.DataFrame):
    """Generate the main dashboard content"""
    # Calculate summary statistics
    total_examples = len(df)
    correct_examples = df['exact_match'].sum()
    accuracy = correct_examples / total_examples if total_examples > 0 else 0
    avg_reward = df['reward'].mean()
    
    # Build comprehensive performance table
    all_rows = []
    
    # Define reward columns to include in table
    reward_columns = [
        ('exact_match_reward', 'EM Reward'),
        ('f1_reward', 'F1 Reward'), 
        ('retrieval_recall_reward', 'Retrieval Reward'),
        ('citation_reward', 'Citation Reward'),
        ('format_reward', 'Format Reward')
    ]
    
    # Overall aggregation row
    total_examples = len(df)
    correct_examples = df['exact_match'].sum()
    overall_accuracy = correct_examples / total_examples if total_examples > 0 else 0
    overall_reward = df['reward'].mean()
    
    # Get available reward columns
    reward_cells = []
    for reward_col, _ in reward_columns:
        if reward_col in df.columns:
            reward_cells.append(Td(f"{df[reward_col].mean():.3f}", cls="px-3 py-2 text-sm text-gray-600"))
        else:
            reward_cells.append(Td("-", cls="px-3 py-2 text-sm text-gray-400"))
    
    all_rows.append(
        Tr(
            Td("All Questions", cls="px-3 py-2 text-sm font-bold text-blue-900 bg-blue-50"),
            Td(f"{total_examples} examples", cls="px-3 py-2 text-sm text-gray-600 bg-blue-50"),
            Td(f"{overall_accuracy:.1%}", cls=f"px-3 py-2 text-sm font-medium bg-blue-50 {'text-green-700' if overall_accuracy >= 0.5 else 'text-red-700'}"),
            Td(f"{overall_reward:.3f}", cls="px-3 py-2 text-sm font-bold text-gray-900 bg-blue-50"),
            *[Td(cell.children[0], cls="px-3 py-2 text-sm text-gray-600 bg-blue-50") for cell in reward_cells]
        )
    )
    
    # Performance by hops
    for hop_count in sorted(df['n_hops'].unique()):
        hop_data = df[df['n_hops'] == hop_count]
        total = len(hop_data)
        correct = hop_data['exact_match'].sum()
        hop_accuracy = correct / total if total > 0 else 0
        hop_reward = hop_data['reward'].mean()
        
        # Reward columns for this hop count
        hop_reward_cells = []
        for reward_col, _ in reward_columns:
            if reward_col in hop_data.columns:
                hop_reward_cells.append(Td(f"{hop_data[reward_col].mean():.3f}", cls="px-3 py-2 text-sm text-gray-600"))
            else:
                hop_reward_cells.append(Td("-", cls="px-3 py-2 text-sm text-gray-400"))
        
        all_rows.append(
            Tr(
                Td(f"{hop_count} hops", cls="px-3 py-2 text-sm font-medium text-gray-900"),
                Td(f"{total} examples", cls="px-3 py-2 text-sm text-gray-600"),
                Td(f"{hop_accuracy:.1%}", cls=f"px-3 py-2 text-sm font-medium {'text-green-700' if hop_accuracy >= 0.5 else 'text-red-700'}"),
                Td(f"{hop_reward:.3f}", cls="px-3 py-2 text-sm text-gray-600"),
                *hop_reward_cells
            )
        )
    
    return Div(
        # Performance & Reward Breakdown
        Div(
            H2("Performance & Reward Breakdown", cls="text-xl font-semibold text-gray-900 mb-4"),
            Div(
                Table(
                    Thead(
                        Tr(
                            Th("Question Type", cls="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase"),
                            Th("Count", cls="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase"),
                            Th("Accuracy", cls="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase"),
                            Th("Overall Reward", cls="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase"),
                            *[Th(display_name, cls="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase") 
                              for reward_col, display_name in reward_columns if reward_col in df.columns],
                        )
                    ),
                    Tbody(*all_rows),
                    cls="min-w-full divide-y divide-gray-200"
                ),
                cls="bg-white shadow-sm rounded-lg overflow-hidden"
            ),
            cls="mb-8"
        ),
        
        # Records List
        Div(
            H2("Inspect Records", cls="text-xl font-semibold text-gray-900 mb-4"),
            simple_records_list(df),
            cls="mb-8"
        ),
        
        cls="space-y-8"
    )

@rt("/example/{idx}")
def example_detail(idx: int):
    """Detailed view of a specific example"""
    if 'raw_results' not in data_store:
        return Div("No data loaded", cls="text-red-600")
    
    raw_results = data_store['raw_results']
    if idx >= len(raw_results):
        return Div("Example not found", cls="text-red-600")
    
    example = raw_results[idx]
    df = data_store['df']
    row = df.iloc[idx]
    
    # Pass the full example including docs
    return Div(
            # Main content
            inspect_view(example, row),
            
            cls="max-w-7xl mx-auto p-6"
        )
    
def is_correct(row: pd.Series) -> bool:
    return row['f1_reward'] >= 0.9

@rt("/filter")
def filter_results(
    correct_only: Optional[str] = None,
    incorrect_only: Optional[str] = None,
    min_hops: Optional[int] = None,
    max_hops: Optional[int] = None,
    min_reward: Optional[float] = None
):
    """Filter results based on criteria"""
    if 'df' not in data_store:
        return Div("No data loaded", cls="text-red-600")
    
    df = data_store['df'].copy()
    df['is_correct'] = df.apply(is_correct, axis=1)
    
    # Apply filters
    if correct_only == "true":
        df = df[df['is_correct'] == 1]
    elif incorrect_only == "true":
        df = df[df['is_correct'] == 0]
    
    if min_hops is not None:
        df = df[df['n_hops'] >= min_hops]
    
    if max_hops is not None:
        df = df[df['n_hops'] <= max_hops]
    
    if min_reward is not None:
        df = df[df['reward'] >= min_reward]
    
    return results_table(df)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8777, reload=True)