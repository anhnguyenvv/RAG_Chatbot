"""Session-based conversation memory backed by MongoDB.

Stores per-session:
  - messages: full conversation history (role + content)
  - summary: LLM-generated summary of older turns
  - context: user context (nganh, khoa, he_dao_tao) extracted from conversation
  - message_count, last_access

MongoDB collection: ``chat_sessions`` with a TTL index on ``expires_at``.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    trim_messages,
)

try:
    from pymongo import MongoClient
except ImportError:
    MongoClient = None  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)


class MongoSessionMemoryStore:
    """MongoDB-backed session store with TTL expiry.

    Each session document schema::

        {
            "_id": <session_id>,
            "messages": [
                {"role": "human", "content": "..."},
                {"role": "ai",    "content": "..."},
            ],
            "summary": "...",
            "context": {
                "nganh": "CNTT",
                "khoa": "K2023",
                "he_dao_tao": "chinh-quy",
            },
            "message_count": 5,
            "created_at": ISODate,
            "last_access": ISODate,
            "expires_at": ISODate,   # TTL index
        }
    """

    def __init__(
        self,
        mongo_uri: str = "mongodb://localhost:27017",
        db_name: str = "rag_chatbot",
        collection_name: str = "chat_sessions",
        max_recent_turns: int = 3,
        max_token_limit: int = 1000,
        session_ttl_seconds: int = 1800,
    ) -> None:
        self.max_recent_turns = max_recent_turns
        self.max_token_limit = max_token_limit
        self.session_ttl_seconds = session_ttl_seconds

        if MongoClient is None:
            raise RuntimeError("pymongo is required: pip install pymongo")

        self._client = MongoClient(mongo_uri)
        self._db = self._client[db_name]
        self._col = self._db[collection_name]

        # TTL index — MongoDB auto-deletes docs when expires_at passes
        self._col.create_index("expires_at", expireAfterSeconds=0)
        # Fast lookup
        self._col.create_index("last_access")

        logger.info(
            "MongoSessionMemoryStore connected: %s/%s.%s (TTL=%ds)",
            mongo_uri, db_name, collection_name, session_ttl_seconds,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _now(self) -> datetime:
        return datetime.now(timezone.utc)

    def _expires_at(self) -> datetime:
        return self._now() + timedelta(seconds=self.session_ttl_seconds)

    def _estimate_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)

    @staticmethod
    def _msg_to_dict(msg: BaseMessage) -> dict[str, str]:
        if isinstance(msg, HumanMessage):
            role = "human"
        elif isinstance(msg, AIMessage):
            role = "ai"
        elif isinstance(msg, SystemMessage):
            role = "system"
        else:
            role = "unknown"
        return {"role": role, "content": str(msg.content)}

    @staticmethod
    def _dict_to_msg(d: dict[str, str]) -> BaseMessage:
        role = d.get("role", "unknown")
        content = d.get("content", "")
        if role == "human":
            return HumanMessage(content=content)
        elif role == "ai":
            return AIMessage(content=content)
        elif role == "system":
            return SystemMessage(content=content)
        return HumanMessage(content=content)

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    def touch_session(self, session_id: str) -> None:
        """Create session if new, or refresh its TTL."""
        now = self._now()
        result = self._col.update_one(
            {"_id": session_id},
            {
                "$set": {
                    "last_access": now,
                    "expires_at": self._expires_at(),
                },
                "$setOnInsert": {
                    "messages": [],
                    "summary": "",
                    "context": {},
                    "message_count": 0,
                    "created_at": now,
                },
            },
            upsert=True,
        )
        if result.upserted_id:
            logger.debug("New session created: %s", session_id)

    def clear_session(self, session_id: str) -> bool:
        result = self._col.delete_one({"_id": session_id})
        return result.deleted_count > 0

    def list_sessions(self) -> list[dict[str, Any]]:
        cursor = self._col.find(
            {},
            {
                "_id": 1,
                "last_access": 1,
                "message_count": 1,
                "summary": 1,
                "context": 1,
            },
        ).sort("last_access", -1).limit(100)

        sessions = []
        for doc in cursor:
            sessions.append({
                "session_id": doc["_id"],
                "last_access": doc.get("last_access", ""),
                "message_count": doc.get("message_count", 0),
                "has_summary": bool(doc.get("summary")),
                "context": doc.get("context", {}),
            })
        return sessions

    # ------------------------------------------------------------------
    # Messages (full conversation history)
    # ------------------------------------------------------------------

    def add_message(self, session_id: str, message: BaseMessage) -> None:
        """Append a message to session history."""
        self._col.update_one(
            {"_id": session_id},
            {
                "$push": {"messages": self._msg_to_dict(message)},
                "$inc": {"message_count": 1},
                "$set": {
                    "last_access": self._now(),
                    "expires_at": self._expires_at(),
                },
            },
        )

    def add_messages(self, session_id: str, messages: list[BaseMessage]) -> None:
        """Append multiple messages at once."""
        if not messages:
            return
        msg_dicts = [self._msg_to_dict(m) for m in messages]
        self._col.update_one(
            {"_id": session_id},
            {
                "$push": {"messages": {"$each": msg_dicts}},
                "$inc": {"message_count": len(messages)},
                "$set": {
                    "last_access": self._now(),
                    "expires_at": self._expires_at(),
                },
            },
        )

    def get_messages(self, session_id: str) -> list[BaseMessage]:
        """Get all messages for a session."""
        doc = self._col.find_one({"_id": session_id}, {"messages": 1})
        if not doc or not doc.get("messages"):
            return []
        return [self._dict_to_msg(d) for d in doc["messages"]]

    def get_recent_messages(self, session_id: str, n: int | None = None) -> list[BaseMessage]:
        """Get the last N messages (default: max_recent_turns * 2)."""
        count = n or self.max_recent_turns * 2
        doc = self._col.find_one(
            {"_id": session_id},
            {"messages": {"$slice": -count}},
        )
        if not doc or not doc.get("messages"):
            return []
        return [self._dict_to_msg(d) for d in doc["messages"]]

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def get_session_summary(self, session_id: str) -> str:
        doc = self._col.find_one({"_id": session_id}, {"summary": 1})
        if not doc:
            return ""
        return doc.get("summary", "")

    def update_summary(self, session_id: str, summary: str) -> None:
        self._col.update_one(
            {"_id": session_id},
            {"$set": {"summary": summary}},
        )

    # ------------------------------------------------------------------
    # User context (nganh, khoa, he_dao_tao)
    # ------------------------------------------------------------------

    def get_context(self, session_id: str) -> dict[str, str]:
        """Get stored user context for this session."""
        doc = self._col.find_one({"_id": session_id}, {"context": 1})
        if not doc:
            return {}
        return doc.get("context", {})

    def update_context(self, session_id: str, context_updates: dict[str, str]) -> None:
        """Merge new context fields into existing context."""
        set_fields = {f"context.{k}": v for k, v in context_updates.items() if v}
        if not set_fields:
            return
        self._col.update_one(
            {"_id": session_id},
            {"$set": set_fields},
        )
        logger.debug("Context updated for session=%s: %s", session_id, context_updates)

    # ------------------------------------------------------------------
    # Message count
    # ------------------------------------------------------------------

    def increment_message_count(self, session_id: str) -> int:
        result = self._col.find_one_and_update(
            {"_id": session_id},
            {
                "$inc": {"message_count": 1},
                "$set": {
                    "last_access": self._now(),
                    "expires_at": self._expires_at(),
                },
            },
            return_document=True,
        )
        if result:
            return result.get("message_count", 0)
        return 0

    # ------------------------------------------------------------------
    # Buffer + Summary hybrid strategy
    # ------------------------------------------------------------------

    def prepare_messages_with_summary(
        self,
        messages: list[BaseMessage],
        summary: str,
    ) -> list[BaseMessage]:
        """Trim messages to keep recent turns and prepend summary."""
        keep_count = self.max_recent_turns * 2
        recent = messages[-keep_count:] if len(messages) > keep_count else messages

        result: list[BaseMessage] = []
        if summary:
            result.append(
                SystemMessage(content=f"Tom tat cuoc hoi thoai truoc do:\n{summary}")
            )
        result.extend(recent)
        return result



    def should_summarize(self, messages: list[BaseMessage]) -> bool:
        """Check if older messages should be summarized."""
        keep_count = self.max_recent_turns * 2
        return len(messages) > keep_count
