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

def inspect_view(example: Dict[str, Any], row: pd.Series) -> Div:
    """Create a simplified trajectory view focusing on key information"""
    
    # All metrics breakdown
    metrics_list = []
    
    # Define all possible metrics and their display names
    metric_definitions = [
        ('reward', 'Overall Reward'),
        ('exact_match', 'Exact Match'),
        ('exact_match_reward', 'Exact Match Reward'),
        ('f1_reward', 'F1 Score Reward'),
        ('retrieval_recall_reward', 'Retrieval Recall Reward'),
        ('citation_reward', 'Citation Reward'),
        ('format_reward', 'Format Reward'),
        ('n_hops', 'Number of Hops'),
        ('n_turns', 'Number of Turns'),
        ('n_tool_calls', 'Number of Tool Calls'),
        ('used_supporting_docs', 'Used Supporting Docs')
    ]
    
    for metric_key, display_name in metric_definitions:
        if metric_key in row and pd.notna(row[metric_key]):
            value = row[metric_key]
            
            # Format the value based on type
            if metric_key == 'exact_match':
                formatted_value = "✅ Correct" if value else "❌ Incorrect"
                value_class = "text-green-700" if value else "text-red-700"
            elif metric_key in ['used_supporting_docs'] and isinstance(value, (int, float)):
                formatted_value = f"{value:.0%}"
                value_class = "text-gray-900"
            elif isinstance(value, (int, float)) and metric_key.endswith('_reward'):
                formatted_value = f"{value:.3f}"
                value_class = "text-gray-900"
            elif isinstance(value, (int, float)):
                if metric_key in ['n_hops', 'n_turns', 'n_tool_calls']:
                    formatted_value = f"{int(value)}"
                else:
                    formatted_value = f"{value:.3f}"
                value_class = "text-gray-900"
            else:
                formatted_value = str(value)
                value_class = "text-gray-900"
            
            # Color code based on metric performance
            bg_class = "bg-blue-50" if metric_key == 'reward' else "bg-gray-50"
            
            metrics_list.append(
                Div(
                    Span(display_name + ":", cls="text-sm font-medium text-gray-600"),
                    Span(formatted_value, cls=f"text-sm font-bold {value_class} ml-2"),
                    cls=f"flex items-center justify-between p-3 {bg_class} rounded"
                )
            )
    
    # Create metrics sidebar
    key_metrics = Div(
        A("← Back to Results", href="/", cls="text-blue-600 hover:text-blue-800 mb-4 inline-block"),
        H3("Metrics & Rewards", cls="text-lg font-semibold text-gray-900 mb-4"),
        Div(
            *metrics_list,
            cls="space-y-2"
        ),
        cls="bg-white rounded-lg border p-4 sticky top-6"
    )
    
    # Question and answer section - simplified
    qa_section = Div(
        Div(
            H3("Question", cls="font-semibold text-gray-900 mb-3"),
            P(row['question'], cls="text-gray-800 bg-gray-50 p-4 rounded-lg mb-4"),
            
            H3("Model's Answer", cls="font-semibold text-gray-900 mb-3"),
            P(row['predicted_answer'], cls="text-gray-800 bg-blue-50 p-4 rounded-lg mb-4"),
            
            H3("Correct Answer", cls="font-semibold text-gray-900 mb-3"),
            P(row['reference_answers'], cls="text-gray-800 bg-green-50 p-4 rounded-lg mb-6"),
            
            cls="bg-white rounded-lg border p-6 mb-6"
        )
    )
    
    # Conversation trajectory - the key inspection element
    conversation_section = format_conversation_section(example['completion'])
    
    # Two-column layout with sticky sidebar
    return Div(
        # Sidebar with metrics
        Div(
            key_metrics,
            cls="w-full lg:w-1/3 lg:pr-6"
        ),
        
        # Main content area (scrollable)
        Div(
            qa_section,
            conversation_section,
            cls="w-full lg:w-2/3"
        ),
        
        cls="flex flex-col lg:flex-row gap-6"
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
    hop_table_rows = []
    for hop_count in sorted(df['n_hops'].unique()):
        hop_data = df[df['n_hops'] == hop_count]
        total = len(hop_data)
        correct = hop_data['exact_match'].sum()
        accuracy = correct / total if total > 0 else 0
        avg_reward = hop_data['reward'].mean()
        
        hop_table_rows.append(
            Tr(
                Td(str(hop_count), cls="px-4 py-3 text-sm font-bold text-gray-900"),
                Td(str(total), cls="px-4 py-3 text-sm text-gray-700"),
                Td(str(correct), cls="px-4 py-3 text-sm text-gray-700"),
                Td(f"{accuracy:.1%}", cls=f"px-4 py-3 text-sm font-medium {'text-green-700' if accuracy >= 0.5 else 'text-red-700'}"),
                Td(f"{avg_reward:.3f}", cls="px-4 py-3 text-sm text-gray-700"),
            )
        )
    
    # Recent examples for quick access
    recent_examples = []
    for idx, row in df.head(5).iterrows():
        status_icon = "✅" if row['exact_match'] else "❌"
        status_color = "text-green-600" if row['exact_match'] else "text-red-600"
        
        recent_examples.append(
            Div(
                Div(
                    Span(status_icon, cls=f"{status_color} text-sm mr-2"),
                    A(
                        row['question'][:60] + ("..." if len(row['question']) > 60 else ""),
                        href=f"/example/{idx}",
                        cls="text-blue-600 hover:text-blue-800 text-sm font-medium"
                    ),
                    cls="flex items-start"
                ),
                Div(
                    Span(f"{row['n_hops']} hops", cls="text-xs text-gray-500 mr-2"),
                    Span(f"Reward: {row['reward']:.3f}", cls="text-xs text-gray-500"),
                    cls="mt-1"
                ),
                cls="p-3 bg-gray-50 rounded-lg hover:bg-gray-100 transition-colors"
            )
        )
    
    return Div(
        # Performance by hops - cleaner table
        Div(
            H2("Performance by Hop Count", cls="text-lg font-semibold mb-4 text-gray-900"),
            Div(
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
                cls="overflow-hidden shadow ring-1 ring-black ring-opacity-5 sm:rounded-lg bg-white"
            ),
            cls="mb-8"
        ),
        
        # Quick access to recent examples
        Div(
            H2("Recent Examples", cls="text-lg font-semibold mb-4 text-gray-900"),
            Div(
                *recent_examples,
                cls="space-y-3"
            ),
            Div(
                A("View All Results →", 
                  onclick="showTab('table')", 
                  cls="inline-flex items-center text-blue-600 hover:text-blue-800 text-sm font-medium cursor-pointer"),
                cls="mt-4"
            ),
            cls="bg-white shadow sm:rounded-lg p-6"
        )
    )

def simple_records_list(df: pd.DataFrame) -> Div:
    """Create a simple, clean list of records for inspection"""
    record_items = []
    
    for idx, row in df.iterrows():
        status_icon = "✅" if row['exact_match'] else "❌"
        status_color = "text-green-600" if row['exact_match'] else "text-red-600"
        
        record_items.append(
            A(
                Div(
                    # Header with status and quick info
                    Div(
                        Div(
                            Span(status_icon, cls=f"{status_color} text-lg mr-3"),
                            Span(f"Record {idx + 1}", cls="font-medium text-gray-900"),
                            cls="flex items-center"
                        ),
                        Div(
                            Span(f"{row['n_hops']} hops", cls="text-sm text-gray-500 mr-3"),
                            Span(f"Reward: {row['reward']:.3f}", cls="text-sm text-gray-500"),
                            cls="flex items-center"
                        ),
                        cls="flex items-center justify-between mb-2"
                    ),
                    
                    # Question preview
                    P(row['question'], cls="text-sm text-gray-700 line-clamp-2"),
                    
                    cls="p-4 bg-white border border-gray-200 rounded-lg hover:border-gray-300 hover:shadow-sm transition-all"
                ),
                href=f"/example/{idx}",
                cls="block mb-3"
            )
        )
    
    return Div(
        *record_items,
        cls="space-y-2"
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