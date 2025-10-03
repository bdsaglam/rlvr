"""Multi-hop QA program for MuSiQue dataset."""

from typing import Literal, Protocol

import dspy
from agents import RunContextWrapper
from pydantic import BaseModel
from vf_musique.rewards import extract_all_retrieved_doc_ids
from vf_musique.tools import ToolContext, make_retrieve_tool


class GenerateSearchQuery(dspy.Signature):
    """Given a multi-hop question and information collected so far, generate a search query
    to find the next piece of information needed to answer the question.
    Focus on entities, dates, or facts that need to be resolved step by step."""

    question: str = dspy.InputField(desc="The multi-hop question to answer")
    collected_info: str = dspy.InputField(desc="Information collected from previous retrieval steps.")
    search_query: str = dspy.OutputField(desc="Search query for the next retrieval step")
    top_n: int = dspy.OutputField(desc="Number of documents to retrieve. 1 <= top_n <= 3")


class KeyInformation(BaseModel):
    info: str
    source_doc_id: str

    def format(self):
        return f"{self.info} [{self.source_doc_id}]"


class ExtractInformation(dspy.Signature):
    """Given a question and retrieved documents, extract the key information
    that helps answer the question or leads to the next retrieval step.
    Focus on entities, relationships, dates, and facts."""

    question: str = dspy.InputField(desc="The multi-hop question to answer")
    documents: str = dspy.InputField(desc="Retrieved documents from search")
    key_informations: list[KeyInformation] = dspy.OutputField(
        desc="Key information(s) extracted from retrieved document(s)"
    )


class DecideInfoCollection(dspy.Signature):
    question: str = dspy.InputField(desc="The multi-hop question to answer")
    all_information: str = dspy.InputField(desc="All information collected during retrieval")
    has_collected_enough_info: bool = dspy.OutputField(desc="Has enough information been collected to answer question?")


class GenerateAnswer(dspy.Signature):
    """Given a multi-hop question and all collected information, provide a concise answer and citations.
    The answer should directly address what the question asks for.
    Be specific and use the exact entities/dates/facts from the documents.
    Cite all documents that support the answer."""

    question: str = dspy.InputField(desc="The multi-hop question to answer")
    all_information: str = dspy.InputField(desc="All information collected during retrieval")
    answer: str = dspy.OutputField(desc="Final answer to the question")
    citations: list[str] = dspy.OutputField(desc="List of document IDs to cite for the answer, e.g. `[4, 9]`")


def get_module_cls(prompt_technique: Literal["predict", "cot"] = "cot") -> type[dspy.Module]:
    if prompt_technique == "predict":
        return dspy.Predict
    if prompt_technique == "cot":
        return dspy.ChainOfThought
    raise ValueError(f"Invalid prompt technique: {prompt_technique}")


class MHQAContextWrapper(RunContextWrapper[ToolContext]): ...


class Retriever(Protocol):
    def __call__(self, wrapper: MHQAContextWrapper, query: str, top_n: int, **kwargs) -> str: ...


class MultiHopQA(dspy.Module):
    """Multi-hop question answering module for MuSiQue."""

    def __init__(
        self,
        *,
        prompt_technique: Literal["predict", "cot"] = "cot",
        retriever: str | Retriever = "hybrid",
        max_iter: int = 5,
    ):
        self.max_iter = max_iter

        # Create modules with typed signatures
        module_cls = get_module_cls(prompt_technique)
        self.generate_query = module_cls(GenerateSearchQuery)
        self.extract_info = module_cls(ExtractInformation)
        self.decide_info_collect = module_cls(DecideInfoCollection)
        self.generate_answer = module_cls(GenerateAnswer)

        # Create the retrieve tool
        if isinstance(retriever, str):
            self.retriever = make_retrieve_tool(retriever, default_top_n=2)
        else:
            self.retriever = retriever

    def forward(self, question: str, docs: list, **kwargs) -> dspy.Prediction:
        """
        Forward pass for multi-hop QA.

        Args:
            question: The multi-hop question to answer
            docs: List of documents available for retrieval
        """

        # Create a context object that holds the documents
        run_context_wrapper = MHQAContextWrapper(context=ToolContext(info=dict(docs=docs)))

        # Initialize collected information and retrieved document IDs
        collected_info = []
        retrieved_doc_ids = []

        for hop_idx in range(self.max_iter):
            query_pred = self.generate_query(
                question=question,
                collected_info=self.format_collected_info(collected_info)
            )

            # Retrieve documents using the MuSiQue retrieve tool
            retrieved_text = self.retriever(
                run_context_wrapper,
                query=query_pred.search_query,
                top_n=max(min(query_pred.top_n, 3), 1),
            )

            # Extract document IDs from retrieved text using the official function
            doc_ids = extract_all_retrieved_doc_ids(retrieved_text)
            for doc_id in doc_ids:
                if doc_id not in retrieved_doc_ids:
                    retrieved_doc_ids.append(doc_id)

            # Extract key information from retrieved documents
            info_pred = self.extract_info(question=question, documents=retrieved_text)
            collected_info.extend(info_pred.key_informations)

            decision_pred = self.decide_info_collect(
                question=question,
                all_information=self.format_collected_info(collected_info),
            )
            if decision_pred.has_collected_enough_info:
                break

        # Generate final answer based on all collected information
        answer_pred: GenerateAnswer = self.generate_answer(
            question=question,
            all_information=self.format_collected_info(collected_info),
        )

        return dspy.Prediction(
            answer=answer_pred.answer,
            collected_info=collected_info,
            retrieved_doc_ids=retrieved_doc_ids,
            citations=answer_pred.citations,
            n_turns=hop_idx + 1,
        )

    def format_collected_info(self, collected_info: list[KeyInformation]) -> str:
        if not collected_info:
            return "No information collected yet"
        return "\n".join([item.format() for item in collected_info])
