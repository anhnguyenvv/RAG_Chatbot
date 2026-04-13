"""ReAct agent using LangGraph for RAG with reasoning and tool use."""

from __future__ import annotations

import uuid
from time import perf_counter
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

from .memory import SessionMemoryStore

SYSTEM_PROMPT = """\
Bạn là trợ lý tư vấn học vụ của Khoa Công Nghệ Thông Tin (FIT), \
trường Đại học Khoa Học Tự Nhiên - Đại học Quốc Gia TP.HCM (HCMUS).

Nhiệm vụ của bạn:
- Trả lời các câu hỏi về chương trình đào tạo, quy chế, đề cương môn học, \
điều kiện tốt nghiệp, tín chỉ, và các thông tin học vụ khác.
- Luôn sử dụng các công cụ tìm kiếm (tools) để tra cứu thông tin trước khi trả lời.
- Chỉ trả lời dựa trên thông tin tìm được. Nếu không tìm thấy, hãy nói rõ.
- Trả lời bằng tiếng Việt, rõ ràng, chính xác.

Quy tắc quan trọng:
1. LUÔN dùng tool qdrant_search để tìm kiếm trước khi trả lời câu hỏi về học vụ.
2. Nếu qdrant_search không đủ thông tin, thử fit_website_search để tìm thêm.
3. Nếu không tìm thấy câu trả lời, hướng dẫn sinh viên liên hệ:
   - Khoa CNTT, Phòng I.54, toà nhà I, 227 Nguyễn Văn Cừ, Q.5, TP.HCM
   - Điện thoại: (028) 62884499
   - Email: info@fit.hcmus.edu.vn
4. Không bịa đặt thông tin. Chỉ trả lời những gì có trong tài liệu.
5. Khi trích dẫn, ghi rõ nguồn tài liệu.
"""


class ReactRAGAgent:
    """ReAct agent that uses tools to search and answer questions."""

    def __init__(
        self,
        llm: Any,
        tools: list[Any],
        memory_store: SessionMemoryStore,
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
            prompt=SYSTEM_PROMPT,
            checkpointer=self.checkpointer,
        )

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
        self.memory_store.touch_session(thread_id)

        config = {"configurable": {"thread_id": thread_id}}

        existing_summary = self.memory_store.get_session_summary(thread_id)

        input_messages: list[BaseMessage] = []
        if existing_summary:
            from langchain_core.messages import SystemMessage
            input_messages.append(
                SystemMessage(content=f"Tóm tắt cuộc hội thoại trước:\n{existing_summary}")
            )
        input_messages.append(HumanMessage(content=query))

        try:
            result = await self.agent.ainvoke(
                {"messages": input_messages},
                config=config,
            )
        except Exception as exc:
            return {
                "result": f"Đã xảy ra lỗi khi xử lý: {exc}",
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
            final_answer = "Không thể tạo câu trả lời. Vui lòng thử lại."

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
        self.memory_store.touch_session(thread_id)

        config = {"configurable": {"thread_id": thread_id}}

        existing_summary = self.memory_store.get_session_summary(thread_id)

        input_messages: list[BaseMessage] = []
        if existing_summary:
            from langchain_core.messages import SystemMessage
            input_messages.append(
                SystemMessage(content=f"Tóm tắt cuộc hội thoại trước:\n{existing_summary}")
            )
        input_messages.append(HumanMessage(content=query))

        try:
            result = self.agent.invoke(
                {"messages": input_messages},
                config=config,
            )
        except Exception as exc:
            return {
                "result": f"Đã xảy ra lỗi khi xử lý: {exc}",
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
            final_answer = "Không thể tạo câu trả lời. Vui lòng thử lại."

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
        except Exception:
            pass

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
        except Exception:
            pass
