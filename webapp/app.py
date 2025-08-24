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
    dashboard_card, results_table, trajectory_view, 
    metric_breakdown, error_analysis_section
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
    return Titled("Multi-Hop QA Results Inspector",
        Div(
            # Header
            Div(
                H1("Multi-Hop QA Evaluation Results", cls="text-3xl font-bold text-gray-900 mb-2"),
                P("Interactive error analysis and trajectory inspection", cls="text-gray-600 mb-6"),
                cls="border-b pb-4 mb-6"
            ),
            
            # File upload section
            Div(
                H2("Load Results File", cls="text-xl font-semibold mb-3"),
                Form(
                    Div(
                        Input(type="file", name="results_file", accept=".jsonl", 
                              cls="block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100"),
                        cls="mb-3"
                    ),
                    Button("Load Data", type="submit", cls="bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700"),
                    action="/upload", method="post", enctype="multipart/form-data", cls="space-y-3"
                ),
                id="upload-section", cls="bg-gray-50 p-4 rounded-lg mb-6"
            ),
            
            # Dashboard content (initially hidden)
            Div(id="dashboard-content", cls="hidden"),
            
            cls="max-w-7xl mx-auto p-6"
        )
    )

@rt("/upload", methods=["POST"])
async def upload_file(results_file: UploadFile):
    """Handle file upload and data loading"""
    if not results_file.filename.endswith('.jsonl'):
        return Div("Please upload a .jsonl file", cls="text-red-600")
    
    # Save uploaded file temporarily
    temp_path = f"/tmp/{results_file.filename}"
    with open(temp_path, "wb") as f:
        content = await results_file.read()
        f.write(content)
    
    # Load and process data
    df, raw_results = load_evaluation_data(temp_path)
    
    if df is None:
        return Div("Error loading file. Please check the format.", cls="text-red-600")
    
    # Store in global data store
    data_store['df'] = df
    data_store['raw_results'] = raw_results
    
    # Return dashboard content
    return get_dashboard_content(df)

def get_dashboard_content(df: pd.DataFrame):
    """Generate the main dashboard content"""
    # Calculate summary statistics
    total_examples = len(df)
    correct_examples = df['exact_match'].sum()
    accuracy = correct_examples / total_examples if total_examples > 0 else 0
    avg_reward = df['reward'].mean()
    
    # Error breakdown by hops
    hop_breakdown = df.groupby('n_hops').agg({
        'exact_match': ['count', 'sum', 'mean'],
        'reward': 'mean'
    }).round(3)
    
    return Div(
        # Summary cards
        Div(
            dashboard_card("Total Examples", str(total_examples), "📊"),
            dashboard_card("Accuracy", f"{accuracy:.1%}", "✅" if accuracy > 0.5 else "❌"),
            dashboard_card("Avg Reward", f"{avg_reward:.3f}", "🎯"),
            dashboard_card("Avg Hops", f"{df['n_hops'].mean():.1f}", "🔗"),
            cls="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6"
        ),
        
        # Navigation tabs
        Div(
            Button("Overview", onclick="showTab('overview')", cls="tab-button active", id="tab-overview"),
            Button("Results Table", onclick="showTab('table')", cls="tab-button", id="tab-table"),
            Button("Error Analysis", onclick="showTab('errors')", cls="tab-button", id="tab-errors"),
            cls="border-b mb-6"
        ),
        
        # Tab content
        Div(
            # Overview tab
            Div(
                metric_breakdown(df),
                id="overview-content", cls="tab-content"
            ),
            
            # Results table tab
            Div(
                results_table(df),
                id="table-content", cls="tab-content hidden"
            ),
            
            # Error analysis tab
            Div(
                error_analysis_section(df),
                id="errors-content", cls="tab-content hidden"
            )
        ),
        
        cls="dashboard-main"
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
    
    return Titled(f"Example {idx + 1}",
        Div(
            # Back button
            A("← Back to Results", href="/", cls="text-blue-600 hover:text-blue-800 mb-4 inline-block"),
            
            # Example header
            Div(
                H1(f"Example {idx + 1}", cls="text-2xl font-bold mb-2"),
                Div(
                    Span("✅ Correct" if row['exact_match'] else "❌ Incorrect", 
                         cls=f"px-3 py-1 rounded-full text-sm font-medium {'bg-green-100 text-green-800' if row['exact_match'] else 'bg-red-100 text-red-800'}"),
                    Span(f"Reward: {row['reward']:.3f}", cls="ml-3 text-gray-600"),
                    Span(f"{row['n_hops']} hops", cls="ml-3 text-gray-600"),
                    cls="mb-4"
                ),
                cls="border-b pb-4 mb-6"
            ),
            
            # Main content
            trajectory_view(example, row),
            
            cls="max-w-6xl mx-auto p-6"
        )
    )

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
    
    # Apply filters
    if correct_only == "true":
        df = df[df['exact_match'] == 1]
    elif incorrect_only == "true":
        df = df[df['exact_match'] == 0]
    
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