"""Reusable UI components for the web app"""

import json
import pandas as pd
from typing import Dict, List, Any
from fasthtml.common import *
from data_loader import get_error_patterns

def dashboard_card(title: str, value: str, icon: str) -> Div:
    """Create a dashboard summary card"""
    return Div(
        Div(
            Div(
                Span(icon, cls="text-2xl"),
                Div(
                    H3(title, cls="text-sm font-medium text-gray-500"),
                    P(value, cls="text-2xl font-bold text-gray-900"),
                    cls="ml-3"
                ),
                cls="flex items-center"
            ),
            cls="p-4"
        ),
        cls="bg-white rounded-lg shadow border"
    )

def results_table(df: pd.DataFrame) -> Div:
    """Create a filterable results table"""
    # Filters
    filters = Div(
        Div(
            Label("Filter:", cls="block text-sm font-medium text-gray-700 mb-1"),
            Div(
                Button("All", onclick="filterResults('')", cls="filter-btn active mr-2"),
                Button("Correct", onclick="filterResults('correct')", cls="filter-btn mr-2"),
                Button("Incorrect", onclick="filterResults('incorrect')", cls="filter-btn mr-2"),
                cls="mb-3"
            ),
            
            Div(
                Label("Hops:", cls="block text-sm font-medium text-gray-700 mb-1"),
                Select(
                    Option("All", value=""),
                    *[Option(str(h), value=str(h)) for h in sorted(df['n_hops'].unique())],
                    name="hop_filter", onchange="applyFilters()", cls="block w-32 border-gray-300 rounded-md"
                ),
                cls="inline-block mr-4"
            ),
            
            Div(
                Label("Min Reward:", cls="block text-sm font-medium text-gray-700 mb-1"),
                Input(type="number", step="0.1", min="0", max="1", name="min_reward", 
                      onchange="applyFilters()", cls="block w-24 border-gray-300 rounded-md"),
                cls="inline-block"
            ),
            
            cls="mb-4"
        ),
        cls="bg-gray-50 p-4 rounded-lg mb-4"
    )
    
    # Table
    table_rows = []
    for idx, row in df.iterrows():
        status_badge = Span(
            "✅" if row['exact_match'] else "❌",
            cls=f"px-2 py-1 rounded-full text-xs font-medium {'bg-green-100 text-green-800' if row['exact_match'] else 'bg-red-100 text-red-800'}"
        )
        
        table_rows.append(
            Tr(
                Td(str(idx + 1), cls="px-4 py-2 text-sm text-gray-900"),
                Td(status_badge, cls="px-4 py-2"),
                Td(
                    A(
                        row['question'][:80] + ("..." if len(row['question']) > 80 else ""),
                        href=f"/example/{idx}",
                        cls="text-blue-600 hover:text-blue-800"
                    ),
                    cls="px-4 py-2 text-sm"
                ),
                Td(f"{row['reward']:.3f}", cls="px-4 py-2 text-sm text-gray-900"),
                Td(str(row['n_hops']), cls="px-4 py-2 text-sm text-gray-900"),
                Td(str(row['n_turns']), cls="px-4 py-2 text-sm text-gray-900"),
                Td(str(row['n_tool_calls']), cls="px-4 py-2 text-sm text-gray-900"),
                Td(f"{row['used_supporting_docs']:.0%}", cls="px-4 py-2 text-sm text-gray-900"),
                cls="hover:bg-gray-50"
            )
        )
    
    table = Div(
        Table(
            Thead(
                Tr(
                    Th("#", cls="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider"),
                    Th("Status", cls="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider"),
                    Th("Question", cls="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider"),
                    Th("Reward", cls="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider"),
                    Th("Hops", cls="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider"),
                    Th("Turns", cls="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider"),
                    Th("Tools", cls="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider"),
                    Th("Docs", cls="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider"),
                )
            ),
            Tbody(*table_rows),
            cls="min-w-full divide-y divide-gray-200"
        ),
        cls="overflow-x-auto shadow ring-1 ring-black ring-opacity-5 md:rounded-lg"
    )
    
    return Div(filters, table)

def trajectory_view(example: Dict[str, Any], row: pd.Series) -> Div:
    """Create a detailed trajectory view of an example"""
    
    # Question and answer section
    qa_section = Div(
        H2("Question & Answer", cls="text-xl font-semibold mb-3"),
        Div(
            Div(
                H3("Question", cls="font-medium text-gray-700 mb-2"),
                P(row['question'], cls="text-gray-900 bg-gray-50 p-3 rounded"),
                cls="mb-4"
            ),
            Div(
                H3("Predicted Answer", cls="font-medium text-gray-700 mb-2"),
                P(row['predicted_answer'], cls="text-gray-900 bg-gray-50 p-3 rounded"),
                cls="mb-4"
            ),
            Div(
                H3("Reference Answers", cls="font-medium text-gray-700 mb-2"),
                P(row['reference_answers'], cls="text-gray-900 bg-gray-50 p-3 rounded"),
                cls="mb-4"
            ),
            cls="bg-white border rounded-lg p-4 mb-6"
        )
    )
    
    # Metrics breakdown
    metrics_section = Div(
        H2("Metrics Breakdown", cls="text-xl font-semibold mb-3"),
        Div(
            *[
                Div(
                    Span(metric.replace('_', ' ').title() + ":", cls="font-medium text-gray-700"),
                    Span(f"{row.get(metric, 0):.3f}", cls="ml-2 text-gray-900"),
                    cls="flex justify-between py-1"
                )
                for metric in ['reward', 'exact_match_reward', 'f1_reward', 'retrieval_recall_reward', 'citation_reward', 'format_reward']
                if metric in row
            ],
            cls="bg-white border rounded-lg p-4 space-y-2 mb-6"
        )
    )
    
    # Documents section
    docs_section = format_documents_section(example['info']['docs'])
    
    # Conversation trajectory
    conversation_section = format_conversation_section(example['completion'])
    
    return Div(
        Div(
            Div(qa_section, cls="lg:col-span-2"),
            Div(metrics_section, docs_section, cls="lg:col-span-1"),
            cls="grid grid-cols-1 lg:grid-cols-3 gap-6"
        ),
        conversation_section
    )

def format_documents_section(docs: List[Dict[str, Any]]) -> Div:
    """Format the documents section"""
    doc_items = []
    for doc in docs:
        is_supporting = doc.get('is_supporting', False)
        badge_cls = "bg-green-100 text-green-800" if is_supporting else "bg-gray-100 text-gray-800"
        badge_text = "Supporting" if is_supporting else "Non-supporting"
        
        doc_items.append(
            Div(
                Div(
                    Span(f"Doc {doc['id']}", cls="font-medium text-gray-900"),
                    Span(badge_text, cls=f"ml-2 px-2 py-1 text-xs rounded-full {badge_cls}"),
                    cls="flex items-center justify-between mb-2"
                ),
                P(doc.get('title', 'No title'), cls="text-sm text-gray-600 mb-2"),
                P(doc.get('body', '')[:200] + ("..." if len(doc.get('body', '')) > 200 else ""), 
                  cls="text-xs text-gray-500"),
                cls="border-l-4 pl-3 mb-3 " + ("border-green-400" if is_supporting else "border-gray-300")
            )
        )
    
    return Div(
        H2("Available Documents", cls="text-xl font-semibold mb-3"),
        Div(
            *doc_items,
            cls="bg-white border rounded-lg p-4 max-h-96 overflow-y-auto"
        )
    )

def format_conversation_section(completion: List[Dict[str, Any]]) -> Div:
    """Format the conversation trajectory"""
    messages = []
    
    for i, message in enumerate(completion):
        role = message.get('role', 'unknown')
        content = message.get('content', '')
        
        if role == 'assistant':
            # Assistant message
            message_div = Div(
                Div(
                    Span("🤖 Assistant", cls="font-medium text-blue-700"),
                    Span(f"Turn {len([m for m in completion[:i+1] if m.get('role') == 'assistant'])}", 
                         cls="text-xs text-gray-500 ml-2"),
                    cls="mb-2"
                ),
                
                # Content
                Div(content, cls="prose prose-sm max-w-none bg-blue-50 p-3 rounded") if content else None,
                
                # Tool calls
                *format_tool_calls(message.get('tool_calls', [])),
                
                cls="mb-4 border-l-4 border-blue-400 pl-4"
            )
            messages.append(message_div)
            
        elif role == 'tool':
            # Tool response
            # Truncate long tool responses
            truncated_content = content[:1000] + "..." if len(content) > 1000 else content
            
            messages.append(
                Div(
                    Div("⚙️ Tool Response", cls="font-medium text-green-700 mb-2"),
                    Pre(truncated_content, cls="text-xs bg-green-50 p-3 rounded overflow-x-auto whitespace-pre-wrap"),
                    cls="mb-4 border-l-4 border-green-400 pl-4"
                )
            )
    
    return Div(
        H2("Conversation Trajectory", cls="text-xl font-semibold mb-3"),
        Div(
            *messages,
            cls="bg-white border rounded-lg p-4"
        )
    )

def format_tool_calls(tool_calls: List[str]) -> List[Div]:
    """Format tool calls"""
    if not tool_calls:
        return []
    
    tool_divs = []
    for tc in tool_calls:
        try:
            if isinstance(tc, str):
                tc_data = json.loads(tc)
                func_name = tc_data.get('function', {}).get('name', 'unknown')
                args = tc_data.get('function', {}).get('arguments', '{}')
                
                tool_divs.append(
                    Div(
                        Div("🔧 Tool Call", cls="font-medium text-purple-700 mb-1"),
                        Div(
                            Span(f"{func_name}(", cls="font-mono text-sm"),
                            Span(args[:100] + ("..." if len(args) > 100 else ""), cls="font-mono text-xs text-gray-600"),
                            Span(")", cls="font-mono text-sm"),
                        ),
                        cls="bg-purple-50 p-2 rounded mb-2"
                    )
                )
        except json.JSONDecodeError:
            tool_divs.append(
                Div("Invalid tool call format", cls="bg-red-50 p-2 rounded mb-2 text-red-700")
            )
    
    return tool_divs

def metric_breakdown(df: pd.DataFrame) -> Div:
    """Create metrics breakdown visualization"""
    # Performance by hops
    hop_performance = df.groupby('n_hops').agg({
        'exact_match': ['count', 'sum', 'mean'],
        'reward': 'mean'
    }).round(3)
    
    hop_table_rows = []
    for hop_count in sorted(df['n_hops'].unique()):
        hop_data = df[df['n_hops'] == hop_count]
        total = len(hop_data)
        correct = hop_data['exact_match'].sum()
        accuracy = correct / total if total > 0 else 0
        avg_reward = hop_data['reward'].mean()
        
        hop_table_rows.append(
            Tr(
                Td(str(hop_count), cls="px-4 py-2 text-sm font-medium text-gray-900"),
                Td(str(total), cls="px-4 py-2 text-sm text-gray-900"),
                Td(str(correct), cls="px-4 py-2 text-sm text-gray-900"),
                Td(f"{accuracy:.1%}", cls="px-4 py-2 text-sm text-gray-900"),
                Td(f"{avg_reward:.3f}", cls="px-4 py-2 text-sm text-gray-900"),
            )
        )
    
    return Div(
        H2("Performance Breakdown", cls="text-xl font-semibold mb-4"),
        
        # Performance by hops table
        Div(
            H3("Performance by Number of Hops", cls="text-lg font-medium mb-3"),
            Table(
                Thead(
                    Tr(
                        Th("Hops", cls="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase"),
                        Th("Total", cls="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase"),
                        Th("Correct", cls="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase"),
                        Th("Accuracy", cls="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase"),
                        Th("Avg Reward", cls="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase"),
                    )
                ),
                Tbody(*hop_table_rows),
                cls="min-w-full divide-y divide-gray-200"
            ),
            cls="bg-white shadow overflow-hidden sm:rounded-lg mb-6"
        ),
        
        # Reward distributions (placeholder for chart)
        Div(
            H3("Reward Distribution", cls="text-lg font-medium mb-3"),
            Div(id="reward-chart", cls="h-64 bg-gray-100 rounded flex items-center justify-center"),
            Canvas(id="rewardChart", width="400", height="200", cls="hidden"),
            cls="bg-white shadow sm:rounded-lg p-4"
        )
    )

def error_analysis_section(df: pd.DataFrame) -> Div:
    """Create error analysis section"""
    patterns = get_error_patterns(df)
    
    # Error summary cards
    error_cards = Div(
        dashboard_card("Total Errors", str(patterns['total_errors']), "❌"),
        dashboard_card("No Answer", str(patterns['no_answer_count']), "❓"),
        dashboard_card("Avg Tools (Correct)", f"{patterns['avg_tools_correct']:.1f}", "🔧"),
        dashboard_card("Avg Tools (Incorrect)", f"{patterns['avg_tools_incorrect']:.1f}", "⚙️"),
        cls="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6"
    )
    
    # Reward comparison
    reward_comparison_rows = []
    for metric, data in patterns['reward_comparison'].items():
        reward_comparison_rows.append(
            Tr(
                Td(metric.replace('_', ' ').title(), cls="px-4 py-2 text-sm font-medium text-gray-900"),
                Td(f"{data['correct']:.3f}", cls="px-4 py-2 text-sm text-green-700"),
                Td(f"{data['incorrect']:.3f}", cls="px-4 py-2 text-sm text-red-700"),
                Td(f"{data['difference']:+.3f}", cls="px-4 py-2 text-sm text-gray-900"),
            )
        )
    
    reward_table = Div(
        H3("Reward Components: Correct vs Incorrect", cls="text-lg font-medium mb-3"),
        Table(
            Thead(
                Tr(
                    Th("Metric", cls="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase"),
                    Th("Correct Avg", cls="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase"),
                    Th("Incorrect Avg", cls="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase"),
                    Th("Difference", cls="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase"),
                )
            ),
            Tbody(*reward_comparison_rows),
            cls="min-w-full divide-y divide-gray-200"
        ),
        cls="bg-white shadow overflow-hidden sm:rounded-lg mb-6"
    )
    
    return Div(
        H2("Error Analysis", cls="text-xl font-semibold mb-4"),
        error_cards,
        reward_table
    )