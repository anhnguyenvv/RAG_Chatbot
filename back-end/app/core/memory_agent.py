"""Memory agent for processing contextual information and summarizing chat history."""

import logging
import re
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from app.config.prompts import MEMORY_SUMMARY_PROMPT_NEW, MEMORY_SUMMARY_PROMPT_EXISTING

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


class MemoryAgent:
    """Sub-agent responsible for extracting context and summarizing chat sessions."""

    def __init__(self, llm: Any, memory_store: Any) -> None:
        self.llm = llm
        self.memory_store = memory_store

    def extract_and_save_context(
        self, thread_id: str, query: str, answer: str
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

    def _build_summary_prompt(self, older_messages: list[BaseMessage], existing_summary: str) -> str:
        """Build the summary prompt using config prompt templates."""
        conversation_text = ""
        for msg in older_messages:
            role = "Nguoi dung" if isinstance(msg, HumanMessage) else "Tro ly"
            conversation_text += f"{role}: {msg.content}\n"

        if existing_summary:
            return MEMORY_SUMMARY_PROMPT_EXISTING.format(
                existing_summary=existing_summary,
                conversation_text=conversation_text
            )
        
        return MEMORY_SUMMARY_PROMPT_NEW.format(
            conversation_text=conversation_text
        )

    def summarize_session_sync(self, thread_id: str, messages: list[BaseMessage]) -> None:
        """Synchronously summarize older messages if conversation is long enough."""
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
        summary_prompt = self._build_summary_prompt(older, existing_summary)

        try:
            summary_response = self.llm.invoke(summary_prompt)
            summary_text = str(getattr(summary_response, "content", summary_response))
            self.memory_store.update_summary(thread_id, summary_text)
            logger.debug("Summary updated for thread=%s len=%d", thread_id, len(summary_text))
        except Exception:
            logger.warning("Failed to summarize conversation thread=%s", thread_id, exc_info=True)

    async def summarize_session_async(self, thread_id: str, messages: list[BaseMessage]) -> None:
        """Asynchronously summarize older messages if conversation is long enough."""
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
        summary_prompt = self._build_summary_prompt(older, existing_summary)

        try:
            summary_response = await self.llm.ainvoke(summary_prompt)
            summary_text = str(getattr(summary_response, "content", summary_response))
            self.memory_store.update_summary(thread_id, summary_text)
            logger.debug("Summary updated for thread=%s len=%d", thread_id, len(summary_text))
        except Exception:
            logger.warning("Failed to summarize conversation thread=%s", thread_id, exc_info=True)
