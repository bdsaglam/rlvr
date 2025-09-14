"""Orchestration tools for integrating sub-agent with verifiers environment."""

from agents import RunContextWrapper

from .sub_agent import RagQaAgent
from .tools import ToolContext


def make_sub_agent_tool(retriever: str = "hybrid", model: str = None):
    """Create a tool that delegates to the sub-agent for document retrieval and answering."""

    # Initialize the sub-agent
    sub_agent = RagQaAgent(retriever=retriever, model=model)

    async def answer_subquestion(ctx: RunContextWrapper[ToolContext], sub_question: str) -> str:
        """
        Delegate a sub-question to the retrieval sub-agent.

        The sub-agent will:
        1. Use document retrieval to find relevant information
        2. Answer the sub-question based on retrieved documents
        3. Provide reasoning and document citations

        Args:
            sub_question: A focused sub-question to be answered using document retrieval.

        Returns:
            The sub-agent's response with reasoning, answer, and cited documents.
        """
        # Simply await the async sub-agent method with context
        try:
            result = await sub_agent.answer_subquestion(sub_question, ctx)
            return result
        except Exception as e:
            return f"Error running sub-agent: {str(e)}"

    return answer_subquestion


def make_planning_tool():
    """Create a tool for the main agent to plan its multi-hop reasoning strategy."""

    async def plan_reasoning(ctx: RunContextWrapper[ToolContext], main_question: str) -> str:
        """
        Plan the multi-hop reasoning strategy for answering the main question.

        This tool helps the main agent think about:
        1. What sub-questions need to be answered
        2. The order in which to ask them
        3. How the answers might connect

        Args:
            main_question: The main question that needs multi-hop reasoning.

        Returns:
            A reasoning plan with suggested sub-questions.
        """
        # For now, this is a simple tool that provides structure
        # In the future, this could be enhanced with more sophisticated planning
        return f"""
**Main Question:** {main_question}

**Planning Guidance:**
1. Break down the main question into 2-3 focused sub-questions
2. Each sub-question should target specific information needed
3. Consider the logical flow: What do you need to know first?
4. Use the answer_subquestion tool for each sub-question
5. Synthesize the sub-answers into a final response

**Next Step:** Identify your first sub-question and use the answer_subquestion tool.
"""

    return plan_reasoning