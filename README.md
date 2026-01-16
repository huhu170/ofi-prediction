# Paper Project（恢复版）

本项目的目标是把 **PDF → Markdown（可检索/可引用）→ 笔记（可追溯/可组织）→ 论文写作** 串成稳定流程。

## 目录结构

- `ref_mineru/`：MinerU 解析后的 Markdown（每篇文献 1 个 `.md`）
- `notes_mineru/`：基于 `ref_mineru/*.md` 生成的文献笔记（强制遵守 `.cursorrules` 的“笔记要求”）
- `tools/`：脚本

## 1）PDF 转 Markdown（MinerU）

脚本：`tools/mineru_batch.py`

默认**优先**在项目内目录搜索 PDF（更可复现，适配从 C 盘迁移到 D 盘）：
- `开题报告文献/`
- `英文文献期刊/`

若项目内目录不存在，才会回退去桌面 `论文/开题报告文献/` 和 `论文/英文文献期刊/`。
也可以用参数显式指定数据源目录（推荐用于严格可追溯）：`--src "D:\...\开题报告文献" --src "D:\...\英文文献期刊"`。

输出：
- 目标 Markdown：`ref_mineru/<paper_id>.md`
- 日志：`ref_mineru/run.log`、`ref_mineru/index.jsonl`
- 中间产物：默认自动删除（`--raw delete`）

在终端运行（项目根目录执行；断点续跑：已生成的 md 会自动跳过）：

```bash
python tools/mineru_batch.py
```

## 2）Markdown 生成笔记（OpenAI / OpenAI-compatible）

脚本：`tools/llm_notes_batch.py`

环境变量（填 URL / API_KEY / MODEL 即可用；Windows 不区分大小写）：
- `URL`：OpenAI 兼容服务的 base url（通常带 `/v1`）
  - 例如：`https://api.openai.com/v1`
  - 例如：`http://127.0.0.1:8000/v1`
- `API_KEY`：密钥
- `MODEL`：模型名

也支持更明确的命名（与上面等价）：
- `OPENAI_BASE_URL` / `OPENAI_URL`
- `OPENAI_API_KEY`
- `OPENAI_MODEL`

输出：
- `notes_mineru/<paper_id>.md`

## 3）恢复论文大纲（从 docx 导出）

脚本：`tools/recover_outline_from_docx.py`

默认读取顺序：
- 优先：项目根目录 `目录大纲.docx`
- 回退：桌面 `论文/目录大纲.docx`

输出：项目根目录 `目录大纲.md`。

---

**重要说明**：笔记与正文写作必须遵守 `.cursorrules` 的“零编造 + 可追溯 + 证据分级”硬性规则。

## 数据来源与引用约定（项目内）

- **解析后的可检索原文**：`ref_mineru/*.md`（建议你在写作/综述中以“@ref_mineru”作为原文引用入口）
- **原始 PDF 定位**：`开题报告文献/`、`英文文献期刊/`（建议你在写作/核验时以“@开题报告文献 / @英文文献期刊”定位原始来源）

> 说明：以上“@xxx”是你的写作与检索约定；脚本层面以目录为准，不会自动生成任何未核验信息。



