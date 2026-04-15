"""ReAct agent using LangGraph for RAG with reasoning and tool use."""

from __future__ import annotations

import logging
import re
import uuid
from time import perf_counter
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

from app.core.prompts import AGENT_SYSTEM_PROMPT
from app.storage.memory import MongoSessionMemoryStore

logger = logging.getLogger(__name__)

# Mapping of keywords → canonical nganh names for context extraction
_NGANH_KEYWORDS: dict[str, str] = {
    "cong nghe thong tin": "Cong nghe thong tin",
    "cntt": "Cong nghe thong tin",
    "he thong thong tin": "He thong thong tin",
    "httt": "He thong thong tin",
    "khoa hoc may tinh": "Khoa hoc may tinh",
    "khmt": "Khoa hoc may tinh",
    "ky thuat phan mem": "Ky thuat phan mem",
    "ktpm": "Ky thuat phan mem",
    "tri tue nhan tao": "Tri tue nhan tao",
    "ttnt": "Tri tue nhan tao",
    "cu nhan tai nang": "Cu nhan tai nang",
}


class ReactRAGAgent:
    """ReAct agent that uses tools to search and answer questions."""

    def __init__(
        self,
        llm: Any,
        tools: list[Any],
        memory_store: MongoSessionMemoryStore,
        max_iterations: int = 5,
    ) -> None:
        self.llm = llm
        self.tools = tools
        self.memory_store = memory_store
        self.max_iterations = max_iterations
        self.checkpointer = MemorySaver()

        self.agent = create_react_agent(
            model=llm,
            tools=tools,
            prompt=AGENT_SYSTEM_PROMPT,
            checkpointer=self.checkpointer,
        )
        logger.info("ReactRAGAgent initialized with %d tools, max_iterations=%d", len(tools), max_iterations)

    def _extract_thought_process(self, messages: list[BaseMessage]) -> list[dict[str, Any]]:
        """Extract the agent's reasoning steps from message history."""
        steps: list[dict[str, Any]] = []
        for msg in messages:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    steps.append({
                        "type": "tool_call",
                        "tool": tc.get("name", "unknown"),
                        "input": tc.get("args", {}),
                    })
            elif msg.type == "tool":
                steps.append({
                    "type": "tool_result",
                    "tool": getattr(msg, "name", "unknown"),
                    "output_preview": str(msg.content)[:500],
                })
        return steps

    def _extract_sources(self, messages: list[BaseMessage]) -> list[dict[str, Any]]:
        """Extract source documents from tool results."""
        sources: list[dict[str, Any]] = []
        for msg in messages:
            if msg.type == "tool" and getattr(msg, "name", "") == "qdrant_search":
                content = str(msg.content)
                doc_blocks = content.split("---")
                for block in doc_blocks:
                    block = block.strip()
                    if block.startswith("[Doc"):
                        lines = block.split("\n", 1)
                        header = lines[0] if lines else ""
                        sources.append({
                            "header": header,
                            "preview": block[:300],
                        })
        return sources

    def _build_input_messages(
        self, query: str, thread_id: str,
    ) -> list[BaseMessage]:
        """Build input messages with summary and stored context."""
        existing_summary = self.memory_store.get_session_summary(thread_id)
        stored_context = self.memory_store.get_context(thread_id)

        input_messages: list[BaseMessage] = []

        if existing_summary:
            input_messages.append(
                SystemMessage(content=f"Tom tat cuoc hoi thoai truoc:\n{existing_summary}")
            )

        # Inject stored user context so agent uses it as metadata filters
        if stored_context:
            ctx_parts: list[str] = []
            if stored_context.get("nganh"):
                ctx_parts.append(f"Nganh: {stored_context['nganh']}")
            if stored_context.get("khoa"):
                ctx_parts.append(f"Khoa: {stored_context['khoa']}")
            if stored_context.get("he_dao_tao"):
                ctx_parts.append(f"He dao tao: {stored_context['he_dao_tao']}")
            if ctx_parts:
                input_messages.append(
                    SystemMessage(
                        content=(
                            "Ngu canh nguoi dung da cung cap:\n"
                            + "\n".join(ctx_parts)
                            + "\nHay su dung cac thong tin nay lam tham so (nganh, khoa, he_dao_tao) "
                            "khi goi tool qdrant_search de loc ket qua chinh xac hon."
                        )
                    )
                )

        input_messages.append(HumanMessage(content=query))
        return input_messages

    def _extract_and_save_context(
        self, thread_id: str, query: str, answer: str,
    ) -> None:
        """Extract nganh/khoa/he_dao_tao from conversation and save to memory."""
        context_updates: dict[str, str] = {}
        combined = f"{query} {answer}".lower()

        # Detect khoa (K2022, K2023, K2024, ...)
        khoa_match = re.search(r"\bk(20\d{2})\b", combined, re.IGNORECASE)
        if khoa_match:
            context_updates["khoa"] = f"K{khoa_match.group(1)}"

        # Detect nganh
        for keyword, nganh_name in _NGANH_KEYWORDS.items():
            if keyword in combined:
                context_updates["nganh"] = nganh_name
                break

        # Detect he dao tao
        if "chinh quy" in combined:
            context_updates["he_dao_tao"] = "chinh quy"
        elif "tu xa" in combined:
            context_updates["he_dao_tao"] = "tu xa"

        if context_updates:
            self.memory_store.update_context(thread_id, context_updates)
            logger.info("Context extracted for thread=%s: %s", thread_id, context_updates)

    async def arun(
        self,
        query: str,
        session_id: str | None = None,
        debug: bool = False,
    ) -> dict[str, Any]:
        """Run the ReAct agent asynchronously."""
        request_id = str(uuid.uuid4())
        start_time = perf_counter()

        thread_id = session_id or f"anon-{request_id}"
        logger.info("Agent arun start request_id=%s thread=%s query_len=%d", request_id, thread_id, len(query))
        self.memory_store.touch_session(thread_id)

        config = {"configurable": {"thread_id": thread_id}}
        input_messages = self._build_input_messages(query, thread_id)

        try:
            result = await self.agent.ainvoke(
                {"messages": input_messages},
                config=config,
            )
        except Exception as exc:
            logger.exception("Agent ainvoke failed request_id=%s", request_id)
            return {
                "result": f"Da xay ra loi khi xu ly: {exc}",
                "source_documents": [],
                "request_id": request_id,
                "confidence": 0.0,
                "error": str(exc),
            }

        all_messages = result.get("messages", [])

        final_answer = ""
        for msg in reversed(all_messages):
            if isinstance(msg, AIMessage) and msg.content and not getattr(msg, "tool_calls", None):
                final_answer = str(msg.content)
                break

        if not final_answer:
            final_answer = "Khong the tao cau tra loi. Vui long thu lai."

        # Persist conversation to MongoDB
        self.memory_store.add_messages(thread_id, [
            HumanMessage(content=query),
            AIMessage(content=final_answer),
        ])
        self._extract_and_save_context(thread_id, query, final_answer)

        self.memory_store.increment_message_count(thread_id)
        await self._maybe_summarize(thread_id, all_messages)

        thought_process = self._extract_thought_process(all_messages) if debug else []
        sources = self._extract_sources(all_messages)

        total_time = perf_counter() - start_time

        tool_call_count = sum(
            1 for msg in all_messages
            if hasattr(msg, "tool_calls") and msg.tool_calls
        )
        confidence = min(0.95, 0.3 + 0.15 * tool_call_count + (0.1 if sources else 0.0))

        output: dict[str, Any] = {
            "result": final_answer,
            "source_documents": sources,
            "request_id": request_id,
            "confidence": confidence,
            "route": "react_agent",
            "needs_clarification": confidence < 0.5,
            "timings": {"total": total_time},
        }

        if debug:
            output["thought_process"] = thought_process
            output["message_count"] = len(all_messages)

        return output

    def run(
        self,
        query: str,
        session_id: str | None = None,
        debug: bool = False,
    ) -> dict[str, Any]:
        """Run the ReAct agent synchronously."""
        request_id = str(uuid.uuid4())
        start_time = perf_counter()

        thread_id = session_id or f"anon-{request_id}"
        logger.info("Agent run start request_id=%s thread=%s query_len=%d", request_id, thread_id, len(query))
        self.memory_store.touch_session(thread_id)

        config = {"configurable": {"thread_id": thread_id}}
        input_messages = self._build_input_messages(query, thread_id)

        try:
            result = self.agent.invoke(
                {"messages": input_messages},
                config=config,
            )
        except Exception as exc:
            logger.exception("Agent invoke failed request_id=%s", request_id)
            return {
                "result": f"Da xay ra loi khi xu ly: {exc}",
                "source_documents": [],
                "request_id": request_id,
                "confidence": 0.0,
                "error": str(exc),
            }

        all_messages = result.get("messages", [])

        final_answer = ""
        for msg in reversed(all_messages):
            if isinstance(msg, AIMessage) and msg.content and not getattr(msg, "tool_calls", None):
                final_answer = str(msg.content)
                break

        if not final_answer:
            logger.warning("Agent produced no final answer request_id=%s", request_id)
            final_answer = "Khong the tao cau tra loi. Vui long thu lai."

        # Persist conversation to MongoDB
        self.memory_store.add_messages(thread_id, [
            HumanMessage(content=query),
            AIMessage(content=final_answer),
        ])
        self._extract_and_save_context(thread_id, query, final_answer)

        self.memory_store.increment_message_count(thread_id)
        self._maybe_summarize_sync(thread_id, all_messages)

        thought_process = self._extract_thought_process(all_messages) if debug else []
        sources = self._extract_sources(all_messages)

        total_time = perf_counter() - start_time

        tool_call_count = sum(
            1 for msg in all_messages
            if hasattr(msg, "tool_calls") and msg.tool_calls
        )
        confidence = min(0.95, 0.3 + 0.15 * tool_call_count + (0.1 if sources else 0.0))

        logger.info(
            "Agent run done request_id=%s time=%.2fs tool_calls=%d confidence=%.2f sources=%d",
            request_id, total_time, tool_call_count, confidence, len(sources),
        )

        output: dict[str, Any] = {
            "result": final_answer,
            "source_documents": sources,
            "request_id": request_id,
            "confidence": confidence,
            "route": "react_agent",
            "needs_clarification": confidence < 0.5,
            "timings": {"total": total_time},
        }

        if debug:
            output["thought_process"] = thought_process
            output["message_count"] = len(all_messages)

        return output

    async def _maybe_summarize(self, thread_id: str, messages: list[BaseMessage]) -> None:
        """Summarize older messages if conversation is long enough."""
        human_ai_messages = [
            m for m in messages
            if isinstance(m, (HumanMessage, AIMessage)) and not getattr(m, "tool_calls", None)
        ]
        if not self.memory_store.should_summarize(human_ai_messages):
            return

        keep_count = self.memory_store.max_recent_turns * 2
        older = human_ai_messages[:-keep_count]
        if not older:
            return

        existing_summary = self.memory_store.get_session_summary(thread_id)
        summary_prompt = self.memory_store.build_summary_prompt(older, existing_summary)

        try:
            summary_response = await self.llm.ainvoke(summary_prompt)
            summary_text = str(getattr(summary_response, "content", summary_response))
            self.memory_store.update_summary(thread_id, summary_text)
            logger.debug("Summary updated for thread=%s len=%d", thread_id, len(summary_text))
        except Exception:
            logger.warning("Failed to summarize conversation thread=%s", thread_id, exc_info=True)

    def _maybe_summarize_sync(self, thread_id: str, messages: list[BaseMessage]) -> None:
        """Synchronous version of _maybe_summarize."""
        human_ai_messages = [
            m for m in messages
            if isinstance(m, (HumanMessage, AIMessage)) and not getattr(m, "tool_calls", None)
        ]
        if not self.memory_store.should_summarize(human_ai_messages):
            return

        keep_count = self.memory_store.max_recent_turns * 2
        older = human_ai_messages[:-keep_count]
        if not older:
            return

        existing_summary = self.memory_store.get_session_summary(thread_id)
        summary_prompt = self.memory_store.build_summary_prompt(older, existing_summary)

        try:
            summary_response = self.llm.invoke(summary_prompt)
            summary_text = str(getattr(summary_response, "content", summary_response))
            self.memory_store.update_summary(thread_id, summary_text)
            logger.debug("Summary updated for thread=%s len=%d", thread_id, len(summary_text))
        except Exception:
            logger.warning("Failed to summarize conversation thread=%s", thread_id, exc_info=True)
