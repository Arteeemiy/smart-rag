from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, TypedDict

from pydantic import BaseModel, Field

from ..config import Settings
from ..schemas.requests import RetrievalOverrides
from .retrieval_service import RetrievalService


@dataclass
class AgenticRetrievalRun:
    query: str
    analysis: dict[str, Any]


@dataclass
class AgenticExecutionTrace:
    retrieval_runs: list[AgenticRetrievalRun] = field(default_factory=list)
    visited_nodes: list[str] = field(default_factory=list)

    def record_retrieval(self, query: str, analysis: dict[str, Any]) -> None:
        self.retrieval_runs.append(AgenticRetrievalRun(query=query, analysis=analysis))

    @property
    def last_run(self) -> AgenticRetrievalRun | None:
        return self.retrieval_runs[-1] if self.retrieval_runs else None


class AgenticState(TypedDict, total=False):
    question: str
    current_question: str
    documents: str
    answer: str
    rewrite_count: int
    retrieve_count: int


class GradeDocuments(BaseModel):
    binary_score: Literal["yes", "no"] = Field(
        description="Relevance score: 'yes' if relevant, or 'no' if not relevant"
    )


class AgenticRAGService:
    """LangGraph agentic RAG implementation aligned with the LangChain tutorial."""

    GRADE_PROMPT = (
        "You are a grader assessing relevance of a retrieved document to a user question.\n"
        "Here is the retrieved document:\n\n{context}\n\n"
        "Here is the user question: {question}\n"
        "If the document contains keyword(s) or semantic meaning related to the user question, "
        "grade it as relevant.\n"
        "Give a binary score 'yes' or 'no' score to indicate whether the document is relevant "
        "to the question."
    )
    REWRITE_PROMPT = (
        "Look at the input and try to reason about the underlying semantic intent / meaning.\n"
        "Here is the initial question:\n-------\n{question}\n-------\n"
        "Formulate an improved question:"
    )
    GENERATE_PROMPT = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer the question. "
        "If you don't know the answer, just say that you don't know. "
        "Use three sentences maximum and keep the answer concise.\n"
        "Question: {question}\nContext: {context}"
    )

    def __init__(self, retrieval_service: RetrievalService, settings: Settings) -> None:
        self._retrieval_service = retrieval_service
        self._settings = settings

    def _build_models(self):
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as error:
            raise RuntimeError("langchain-openai is required for agentic RAG mode") from error
        if not self._settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is required for agentic RAG mode")

        kwargs = {
            "api_key": self._settings.openai_api_key,
            "model": self._settings.agentic_rag_model,
            "temperature": 0,
            "timeout": self._settings.llm_timeout_seconds,
            "max_retries": self._settings.llm_max_retries,
        }
        if self._settings.openai_api_url:
            kwargs["base_url"] = self._settings.openai_api_url
        return ChatOpenAI(**kwargs), ChatOpenAI(**kwargs)

    def run(
        self,
        question: str,
        *,
        top_k: int | None = None,
        retrieval_overrides: RetrievalOverrides | None = None,
        include_answer: bool = True,
        include_context: bool = True,
        include_retrieved_text: bool = True,
        vector_store_override: Any | None = None,
        retrieval_scope: str = "knowledge_base",
    ) -> dict[str, Any]:
        from langgraph.graph import END, START, StateGraph

        response_model, grader_model = self._build_models()
        effective_top_k = top_k or self._settings.top_k_results
        trace = AgenticExecutionTrace()

        def retrieve(state: AgenticState) -> AgenticState:
            current_question = state.get("current_question") or state.get("question") or question
            analysis = self._retrieval_service.analyze_retrieval(
                query=current_question,
                top_k=effective_top_k,
                overrides=retrieval_overrides,
                vector_store_override=vector_store_override,
                retrieval_scope=retrieval_scope,
            )
            trace.record_retrieval(query=current_question, analysis=analysis)
            return {
                "documents": analysis.get("context") or "",
                "retrieve_count": int(state.get("retrieve_count", 0)) + 1,
            }

        def grade_documents(state: AgenticState) -> Literal["generate_answer", "rewrite_question", END]:
            docs = state.get("documents") or ""
            current_question = state.get("current_question") or state.get("question") or question
            last_run = trace.last_run
            diagnostics = (last_run.analysis or {}).get("diagnostics", {}) if last_run else {}
            if not diagnostics.get("accepted") and int(state.get("rewrite_count", 0)) >= 1:
                return END
            if int(state.get("retrieve_count", 0)) >= 2:
                return "generate_answer" if diagnostics.get("accepted") else END
            prompt = self.GRADE_PROMPT.format(question=current_question, context=docs)
            response = grader_model.with_structured_output(GradeDocuments).invoke(
                [{"role": "user", "content": prompt}]
            )
            if response.binary_score == "yes":
                return "generate_answer"
            if int(state.get("rewrite_count", 0)) >= 1:
                return END
            return "rewrite_question"

        def rewrite_question(state: AgenticState) -> AgenticState:
            current_question = state.get("current_question") or state.get("question") or question
            prompt = self.REWRITE_PROMPT.format(question=current_question)
            response = response_model.invoke([{"role": "user", "content": prompt}])
            return {
                "current_question": response.content,
                "rewrite_count": int(state.get("rewrite_count", 0)) + 1,
            }

        def generate_answer(state: AgenticState) -> AgenticState:
            current_question = state.get("current_question") or state.get("question") or question
            docs = state.get("documents") or ""
            prompt = self.GENERATE_PROMPT.format(question=current_question, context=docs)
            response = response_model.invoke([{"role": "user", "content": prompt}])
            return {"answer": response.content}

        workflow = StateGraph(AgenticState)
        workflow.add_node("retrieve", retrieve)
        workflow.add_node("rewrite_question", rewrite_question)
        workflow.add_node("generate_answer", generate_answer)
        workflow.add_edge(START, "retrieve")
        workflow.add_conditional_edges("retrieve", grade_documents)
        workflow.add_edge("rewrite_question", "retrieve")
        workflow.add_edge("generate_answer", END)
        graph = workflow.compile()

        final_state = graph.invoke(
            {"question": question, "current_question": question, "rewrite_count": 0, "retrieve_count": 0},
            config={"recursion_limit": min(self._settings.agentic_rag_recursion_limit, 6)},
        )

        final_answer = final_state.get("answer")
        messages = []
        return {"messages": [response]}

        def grade_documents(state: MessagesState) -> Literal["generate_answer", "rewrite_question"]:
            user_question = next(
                (
                    message.content
                    for message in state["messages"]
                    if isinstance(message, HumanMessage) and getattr(message, "content", None)
                ),
                question,
            )
            context = state["messages"][-1].content
            prompt = self.GRADE_PROMPT.format(question=user_question, context=context)
            response = grader_model.with_structured_output(GradeDocuments).invoke(
                [{"role": "user", "content": prompt}]
            )
            return "generate_answer" if response.binary_score == "yes" else "rewrite_question"

        def rewrite_question(state: MessagesState):
            user_question = next(
                (
                    message.content
                    for message in state["messages"]
                    if isinstance(message, HumanMessage) and getattr(message, "content", None)
                ),
                question,
            )
            prompt = self.REWRITE_PROMPT.format(question=user_question)
            response = response_model.invoke([{"role": "user", "content": prompt}])
            return {"messages": [HumanMessage(content=response.content)]}

        def generate_answer(state: MessagesState):
            user_question = next(
                (
                    message.content
                    for message in state["messages"]
                    if isinstance(message, HumanMessage) and getattr(message, "content", None)
                ),
                question,
            )
            context = state["messages"][-1].content
            prompt = self.GENERATE_PROMPT.format(question=user_question, context=context)
            response = response_model.invoke([{"role": "user", "content": prompt}])
            return {"messages": [response]}

        workflow = StateGraph(MessagesState)
        workflow.add_node("generate_query_or_respond", generate_query_or_respond)
        workflow.add_node("retrieve", ToolNode([retrieve_documents]))
        workflow.add_node("rewrite_question", rewrite_question)
        workflow.add_node("generate_answer", generate_answer)
        workflow.add_edge(START, "generate_query_or_respond")
        workflow.add_conditional_edges(
            "generate_query_or_respond",
            tools_condition,
            {"tools": "retrieve", END: END},
        )
        workflow.add_conditional_edges("retrieve", grade_documents)
        workflow.add_edge("generate_answer", END)
        workflow.add_edge("rewrite_question", "generate_query_or_respond")
        graph = workflow.compile()

        final_state: dict[str, Any] | None = None
        for chunk in graph.stream(
            {"messages": [{"role": "user", "content": question}]},
            config={"recursion_limit": self._settings.agentic_rag_recursion_limit},
        ):
            for node_name, update in chunk.items():
                trace.visited_nodes.append(node_name)
                final_state = update

        messages = list((final_state or {}).get("messages") or [])
        final_answer = None
        for message in reversed(messages):
            if isinstance(message, AIMessage) and not getattr(message, "tool_calls", None):
                final_answer = message.content
                break

        last_run = trace.last_run
        analysis = last_run.analysis if last_run else None
        has_context = bool(analysis and analysis.get("diagnostics", {}).get("accepted"))
        if include_answer and not final_answer:
            final_answer = (
                "К сожалению, в моей базе знаний нет информации по этому вопросу. Обратитесь к технической поддержке."
                if not has_context
                else "Извините, возникла техническая ошибка при обработке запроса."
            )

        retrieval_diagnostics = analysis.get("diagnostics") if analysis else {
            "accepted": False,
            "reason": "no_retrieval",
            "requested_top_k": effective_top_k,
            "total_hits": 0,
            "selected_hits": 0,
            "retrieval_scope": retrieval_scope,
        }
        retrieval_diagnostics = {
            **retrieval_diagnostics,
            "agentic_trace": {
                "visited_nodes": trace.visited_nodes,
                "retrieval_queries": [item.query for item in trace.retrieval_runs],
            },
        }

        response_messages: list[dict[str, Any]] = []

        return {
            "query": question,
            "answer": final_answer if include_answer else None,
            "has_context": has_context,
            "error": None,
            "sources": analysis.get("sources") if analysis else [],
            "context": analysis.get("context") if include_context and analysis else None,
            "retrieved": analysis.get("retrieved") if include_retrieved_text and analysis else None,
            "retrieval_diagnostics": retrieval_diagnostics,
            "hit_details": analysis.get("hit_details") if analysis else [],
            "query_expansion": {"enabled": False, "variants": [question]},
            "reranking": {"enabled": False, "applied": False},
            "rag_mode": "agentic",
            "response_messages": response_messages,
        }
