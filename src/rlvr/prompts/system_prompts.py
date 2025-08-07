SIMPLE_PROMPT = """
Respond in the following format, using careful step-by-step reasoning.

<think>
...
</think>
<answer>
...
</answer>
"""

CODE_PROMPT = """\
Given a math problem, use step-by-step reasoning and code execution to solve the problem. 

For each step:
1. Think through your reasoning inside <think> tags
2. Write Python scripts inside <code> tags to work out calculations
   - Functions and variables do not persist across <code> calls and should be redefined each time
   - Scripts should be written in Python 3.10+ syntax, and should run in under 10 seconds
   - Any desired outputs should be printed using print() statements
   - You may import numpy, scipy, and sympy libraries for your calculations
3. You will see the output from print() statements in your code in <output> tags
4. Continue until you can give the final answer inside <answer> tags
"""

DEFAULT_TOOL_PROMPT_TEMPLATE = """\
For each step:
1. Think through your reasoning inside <think> tags
2. If needed, use a tool by writing a JSON command inside <tool> tags with:
   - "name": the tool to use
   - "args": the arguments for the tool
For instance,
<tool>
{
  "name": "search",
  "args": {
    "query": "..."
  }
}
</tool>
3. You will see the tool's output inside <result> tags
4. Continue until you can give the final answer
5. At the end, reflect on your previous steps inside <think> tag and give your final answer inside <answer> tag

You have access to the following tools to help solve problems:
{tool_descriptions}

Tools expect specific JSON input formats. Do not make up tools or arguments that aren't listed.
"""


QA_TOOL_PROMPT_TEMPLATE = """\
Answer the question based on the information provided by tools. You have access to the following tools:
====
{tool_descriptions}
====

For each step:
1. Think through your reasoning inside <think> tags
2. If needed, use a tool by writing a JSON command inside <tool> tags with:
   - "name": the tool to use
   - "args": the arguments for the tool
For instance,
<tool>
{{
  "name": "retrieve",
  "args": {{
    "query": "..."
  }}
}}
</tool>
3. You will see the tool's output inside <result> tags
4. Continue until you have found the answer
5. In the **last** step, 
  - Reflect on your previous steps inside <think> tags
  - Cite the documents you used to answer the question inside <cite> tags by their IDs, e.g. `<cite>1, 2, 3</cite>`. This will be used to determine whether you back your answer with the information provided by the tools.
  - Give your final answer inside <answer> tags

- Tools expect specific JSON input formats.
- Do not make up tools or arguments that aren't listed in the tool descriptions.
- If the `retrieve` tool doesn't return a relevant document, try different queries. Your answer must be based solely on retrieved documents and the question is definitely answerable. Continue searching until you find the relevant information.
"""


WIKI_QA_TOOL_PROMPT_TEMPLATE = """\
Answer the question based on the information provided by tools. You have access to the following tools:
====
{tool_descriptions}
====

For each step:
1. Think through your reasoning inside <think> tags
2. If needed, use a tool by writing a JSON command inside <tool> tags with:
   - "name": the tool to use
   - "args": the arguments for the tool
For instance,
<tool>
{{
  "name": "search",
  "args": {{
    "query": "..."
  }}
}}
</tool>
3. You will see the tool's output inside <result> tags
4. Continue until you have found the answer
5. In the **last** step, 
  - Reflect on your previous steps inside <think> tags
  - Cite the documents you used to answer the question inside <cite> tags by their IDs, e.g. `<cite>1, 2, 3</cite>`. This will be used to determine whether you back your answer with the information provided by the tools.
  - Give your final answer inside <answer> tags

- Tools expect specific JSON input formats.
- Do not make up tools or arguments that aren't listed in the tool descriptions.
- If the `retrieve` tool doesn't return a relevant document, try different queries. Your answer must be based solely on retrieved documents and the question is definitely answerable. Continue searching until you find the relevant information.
"""
