# Agent Prompt Templates — RAG Chatbot FIT-HCMUS

Bộ 3 prompt template cho Lead Agent điều phối team coding bằng sub-agent.

## Cách dùng

1. Lead Agent xác định task → chọn template phù hợp (Frontend / Backend / Tester).
2. Copy toàn bộ nội dung file `.md` tương ứng.
3. Điền vào các placeholder `{{...}}` (task, acceptance criteria, file liên quan, API contract...).
4. Gọi sub-agent: `Agent({ prompt: "<nội dung đã điền>", subagent_type: "general-purpose" })`.
5. Nhận output theo format chuẩn → dùng làm input cho agent kế tiếp.

## Quy trình điều phối chuẩn

```
User request
   ↓
[Lead] Spec + impact analysis
   ↓
 ┌─ backend-agent.md ─┐   (song song nếu task độc lập)
 └─ frontend-agent.md ┘
   ↓
tester-agent.md  (nhận output của cả 2)
   ↓
[Lead] Verdict → merge hoặc loop lại
```

## Files

| File | Dành cho | Lint tool |
|---|---|---|
| `frontend-agent.md` | React/Vite/Tailwind UI | eslint |
| `backend-agent.md` | FastAPI/LangGraph/Qdrant | ruff |
| `tester-agent.md` | pytest + smoke + lint check | — |

## Nguyên tắc chung cho mọi template

- Sub-agent **không có memory** — prompt phải self-contained.
- Mỗi template đã embed anti-patterns từ `CLAUDE.md` — KHÔNG xoá khi customize.
- Output format cố định để Lead parse dễ và handoff giữa các agent mượt.
- Quality gate cuối luôn do Tester Agent quyết.

*Last updated: 2026-04-21*
