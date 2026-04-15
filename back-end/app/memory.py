
from __future__ import annotations

import time
from threading import Lock
from typing import Any

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    trim_messages,
)


class SessionMemoryStore:
    """In-memory session store with TTL-based cleanup.

    Tracks session metadata and provides message trimming with
    summarization for the buffer+summary hybrid strategy.
    """

    def __init__(
        self,
        max_recent_turns: int = 3,
        max_token_limit: int = 1000,
        session_ttl_seconds: int = 1800,
    ) -> None:
        self.max_recent_turns = max_recent_turns
        self.max_token_limit = max_token_limit
        self.session_ttl_seconds = session_ttl_seconds
        self._sessions: dict[str, dict[str, Any]] = {}
        self._lock = Lock()

    def _estimate_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)

    def _cleanup_expired(self) -> None:
        now = time.time()
        expired = [
            sid
            for sid, meta in self._sessions.items()
            if now - meta["last_access"] > self.session_ttl_seconds
        ]
        for sid in expired:
            del self._sessions[sid]

    def touch_session(self, session_id: str) -> None:
        with self._lock:
            self._cleanup_expired()
            if session_id not in self._sessions:
                self._sessions[session_id] = {
                    "last_access": time.time(),
                    "summary": "",
                    "message_count": 0,
                }
            else:
                self._sessions[session_id]["last_access"] = time.time()

    def get_session_summary(self, session_id: str) -> str:
        with self._lock:
            meta = self._sessions.get(session_id)
            if meta is None:
                return ""
            return meta.get("summary", "")

    def update_summary(self, session_id: str, summary: str) -> None:
        with self._lock:
            if session_id in self._sessions:
                self._sessions[session_id]["summary"] = summary

    def increment_message_count(self, session_id: str) -> int:
        with self._lock:
            if session_id in self._sessions:
                self._sessions[session_id]["message_count"] += 1
                return self._sessions[session_id]["message_count"]
            return 0

    def clear_session(self, session_id: str) -> bool:
        with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                return True
            return False

    def list_sessions(self) -> list[dict[str, Any]]:
        with self._lock:
            self._cleanup_expired()
            return [
                {
                    "session_id": sid,
                    "last_access": meta["last_access"],
                    "message_count": meta["message_count"],
                    "has_summary": bool(meta.get("summary")),
                }
                for sid, meta in self._sessions.items()
            ]

    def prepare_messages_with_summary(
        self,
        messages: list[BaseMessage],
        summary: str,
    ) -> list[BaseMessage]:
        """Trim messages to keep recent turns and prepend summary of older context.

        Strategy: keep the last `max_recent_turns * 2` messages (human+ai pairs),
        and if there's a summary from older turns, prepend it as a system message.
        """
        keep_count = self.max_recent_turns * 2
        recent = messages[-keep_count:] if len(messages) > keep_count else messages

        result: list[BaseMessage] = []
        if summary:
            result.append(
                SystemMessage(
                    content=f"Tóm tắt cuộc hội thoại trước đó:\n{summary}"
                )
            )
        result.extend(recent)
        return result

    def build_summary_prompt(self, older_messages: list[BaseMessage], existing_summary: str) -> str:
        """Build a prompt for the LLM to summarize older conversation turns."""
        conversation_text = ""
        for msg in older_messages:
            role = "user" if isinstance(msg, HumanMessage) else "assistant"
            conversation_text += f"{role}: {msg.content}\n"

        if existing_summary:
            return (
                f"Dưới đây là tóm tắt cuộc hội thoại trước đó:\n{existing_summary}\n\n"
                f"Và đây là phần hội thoại mới cần tóm tắt thêm:\n{conversation_text}\n\n"
                "Hãy tóm tắt ngắn gọn toàn bộ cuộc hội thoại trên bằng tiếng Việt, "
                "giữ lại các thông tin quan trọng (ngành học, môn học, niên khóa, chương trình đào tạo, điều kiện được hỏi)."
            )

        return (
            f"Hãy tóm tắt ngắn gọn cuộc hội thoại sau bằng tiếng Việt, "
            f"giữ lại các thông tin quan trọng:\n{conversation_text}"
        )

    def should_summarize(self, messages: list[BaseMessage]) -> bool:
        """Check if older messages should be summarized."""
        keep_count = self.max_recent_turns * 2
        return len(messages) > keep_count
