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
[Lead] Fill TASK_SPEC_TEMPLATE.md  ← spec rõ hay mù phụ thuộc bước này
   ↓
[Lead] Impact analysis (gitnexus_impact) → chia task
   ↓
 ┌─ backend-agent.md ─┐   (song song nếu task độc lập)
 └─ frontend-agent.md ┘
   ↓
   Agent tự chạy REVIEW_CYCLE Gate 1 (self-review)
   ↓
[Lead] REVIEW_CYCLE Gate 2 (peer review)
   ↓
tester-agent.md       — REVIEW_CYCLE Gate 3 (test review)
   ↓
[Lead] REVIEW_CYCLE Gate 4 (merge decision)
   ↓
   PASS → Append AGENT_LOG.md → Done
   FAIL → Loop lại gate tương ứng, tạo follow-up task
```

## Files

### Agent templates
| File | Dành cho | Lint tool |
|---|---|---|
| `frontend-agent.md` | React/Vite/Tailwind UI | eslint |
| `backend-agent.md` | FastAPI/LangGraph/Qdrant | ruff |
| `tester-agent.md` | pytest + smoke + lint check | — |

### Workflow docs
| File | Vai trò |
|---|---|
| `TASK_SPEC_TEMPLATE.md` | **Input** — Lead fill trước khi spawn sub-agent |
| `REVIEW_CYCLE.md` | **Output** — 4 gate review sau khi agent báo done |
| `AGENT_LOG.md` | **Audit trail** — features, bugs, blockers team đã xử lý |

## Nguyên tắc chung cho mọi template

- Sub-agent **không có memory** — prompt phải self-contained.
- Mỗi template đã embed anti-patterns từ `CLAUDE.md` — KHÔNG xoá khi customize.
- Output format cố định để Lead parse dễ và handoff giữa các agent mượt.
- Quality gate cuối luôn do Tester Agent quyết.
- **Mọi sub-agent khi đóng task đều phải append 1 entry vào `AGENT_LOG.md`** (Lead chèn giúp nếu agent không tự làm) — format: `YYYY-MM-DD | <agent> | FEAT|FIX|CHORE|BLOCK|PEND | Summary`. Xem legend trong `AGENT_LOG.md`.

*Last updated: 2026-04-21*
