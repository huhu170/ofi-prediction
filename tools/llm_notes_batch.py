from __future__ import annotations

"""
批量用 OpenAI / OpenAI-compatible Chat Completions API 读取 ref_mineru/*.md，
让大模型直接输出最终的“文献笔记 Markdown”，并写入 notes_mineru/*.md。

本脚本刻意保持“本地处理最小化”：只负责
- 读取输入 Markdown
- 调用大模型生成最终 Markdown
- 原样写文件到 notes_mineru/

安全提醒：
- 不要把 API Key 写进代码或命令行历史。
- 推荐把配置放到项目根目录 `.apikey.env`（KEY=VALUE），脚本启动会自动读取。

环境变量/`.apikey.env` 字段（Windows 不区分大小写）：
- URL：OpenAI 兼容服务的 base url（通常带 /v1）
- API_KEY：密钥
- MODEL：模型名

也兼容更明确命名：
- OPENAI_BASE_URL / OPENAI_URL
- OPENAI_API_KEY
- OPENAI_MODEL
"""

import argparse
import io
import json
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import re


REPO_ROOT = Path(__file__).resolve().parents[1]


def _setup_utf8_stdio() -> None:
    # Windows 下避免 GBK 编码导致的 UnicodeEncodeError
    if sys.platform == "win32":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")


def _read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="replace")


def _load_noterules() -> str:
    """
    读取项目根目录 noterules.md（笔记规则），用于注入提示词。
    若文件不存在或读取失败则返回空字符串。
    """
    p = (REPO_ROOT / "noterules.md").resolve()
    if not p.exists():
        return ""
    try:
        return _read_text(p).strip()
    except Exception:
        return ""


def _env_first(*names: str) -> str:
    for n in names:
        v = (os.environ.get(n) or "").strip()
        if v:
            return v
    return ""


def _strip_quotes(v: str) -> str:
    v = (v or "").strip()
    if len(v) >= 2 and ((v[0] == v[-1] == '"') or (v[0] == v[-1] == "'")):
        return v[1:-1]
    return v


def _load_env_file(env_path: Path, override: bool = True) -> None:
    """
    读取一个极简 .env 文件（每行 KEY=VALUE，可带引号，可用 # 注释），写入 os.environ。
    """
    if not env_path.exists() or not env_path.is_file():
        return
    try:
        text = env_path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return
    for ln in text.splitlines():
        s = ln.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        k, v = s.split("=", 1)
        k = k.strip()
        v = _strip_quotes(v.strip())
        if not k:
            continue
        if override or (k not in os.environ):
            os.environ[k] = v


@dataclass
class OpenAIConfig:
    api_key: str
    model: str
    base_url: str  # should include /v1 for most servers
    timeout_s: int = 120
    temperature: float = 0.0


def _tail(s: str, n: int = 4000) -> str:
    return s[-n:] if s else ""


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _safe_filename(name: str) -> str:
    """
    Windows 文件名安全化（保留中文/英文/数字/常见符号）。
    """
    s = (name or "").strip().replace("\u0000", "")
    s = re.sub(r"[<>:\"/\\\\|?*]+", "_", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = s.rstrip(" .")
    return s or "untitled"


def _one_line(s: str, limit: int = 2000) -> str:
    """
    把可能含换行/制表符的字符串压成单行，便于写 jsonl。
    """
    t = (s or "").replace("\r\n", "\n").replace("\r", "\n")
    t = t.replace("\n", "\\n").replace("\t", "\\t")
    t = t.strip()
    if len(t) > limit:
        t = t[:limit] + "…(truncated)"
    return t


def _append_jsonl(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _extract_note_block(paper_id: str, text: str) -> str:
    """
    从模型输出中提取 BEGIN_NOTE/END_NOTE 之间的内容。
    若不存在标记，则尝试从 `# <paper_id>` 所在行开始截取；仍找不到则返回原始 text（trim）。
    """
    s = (text or "")
    m = re.search(r"-----BEGIN_NOTE-----\s*(.*?)\s*-----END_NOTE-----", s, flags=re.S)
    if m:
        return m.group(1).strip()

    # 兜底：从 '# <paper_id>' 的顶级标题开始截断，忽略前置“思考过程”等噪声
    pat = re.compile(rf"^#\s+{re.escape(paper_id)}\s*$", flags=re.M)
    m2 = pat.search(s)
    if m2:
        return s[m2.start() :].strip()

    # 再兜底：从第一条 '# ' 顶级标题开始
    m3 = re.search(r"^#\s+.+$", s, flags=re.M)
    if m3:
        return s[m3.start() :].strip()
    return s.strip()


def _openai_chat_completions_http(cfg: OpenAIConfig, messages: list[dict]) -> str:
    base = cfg.base_url.rstrip("/")
    url = f"{base}/chat/completions"
    payload = {
        "model": cfg.model.strip(),
        "temperature": cfg.temperature,
        "messages": messages,
    }
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {cfg.api_key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=cfg.timeout_s) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace") if hasattr(e, "read") else ""
        raise RuntimeError(f"HTTPError {e.code}: {_tail(body)}")
    except Exception as e:
        raise RuntimeError(f"{type(e).__name__}: {e}")
    obj = json.loads(raw)
    return obj["choices"][0]["message"]["content"] or ""


def _openai_chat_completions_sdk(cfg: OpenAIConfig, messages: list[dict]) -> str:
    """
    使用 OpenAI Python SDK（openai>=1.0）。若未安装 openai，则抛 ImportError。
    """
    from openai import OpenAI  # type: ignore

    client = OpenAI(base_url=cfg.base_url, api_key=cfg.api_key)
    resp = client.chat.completions.create(model=cfg.model, temperature=cfg.temperature, messages=messages)
    return resp.choices[0].message.content or ""


def _llm_generate(cfg: OpenAIConfig, messages: list[dict]) -> str:
    # 优先 SDK（对部分兼容平台更稳），无 SDK 则回退 http
    try:
        return _openai_chat_completions_sdk(cfg, messages)
    except ImportError:
        return _openai_chat_completions_http(cfg, messages)


def _retry(fn, tries: int = 5, base_sleep: float = 2.0) -> str:
    last_err: Exception | None = None
    for k in range(tries):
        try:
            return fn()
        except Exception as e:
            last_err = e
            time.sleep(base_sleep * (2**k))
    raise RuntimeError(f"LLM 调用重试失败：{last_err}")


def _mk_messages(paper_id: str, source_md: str) -> list[dict]:
    noterules = _load_noterules()
    system = (
        "你是一个严谨的“论文写作与文献笔记助手”。\n"
        "必须遵守：零编造；每条重要结论必须附带原文关键句（逐字短引，可在原文中搜索到）。\n"
        "必须严格遵守用户提供的 noterules.md（笔记规则）。\n"
        "输出必须是 Markdown，不要输出 JSON，不要输出代码块。\n"
        "你必须严格按用户给定的输出模板填写；不要添加任何其他章节或前言。\n"
    )

    # 额外给一个“必须照抄标题/字段”的硬模板，避免模型忽略 noterules.md 的结构要求
    template = (
        "-----BEGIN_NOTE-----\n"
        f"# {paper_id}\n\n"
        "## 1）元数据卡（Metadata Card）\n"
        "- 标题：\n"
        "- 作者：\n"
        "- 年份：\n"
        "- 期刊/会议：\n"
        "- DOI/URL：\n"
        "- 适配章节（映射到论文大纲，写 1–3 个）：\n"
        "- 一句话可用结论（必须含证据编号）：\n"
        "- 可复用证据（列出最关键 3–5 条 E 编号）：\n"
        "- 市场/资产（指数/个股/期货/加密等）：\n"
        "- 数据来源（交易所/数据库/公开数据集名称）：\n"
        "- 频率（tick/quote/trade/分钟/日等）：\n"
        "- 预测目标（方向/收益/价格变化/波动/冲击等）：\n"
        "- 预测视角（点预测/区间/分类/回归）：\n"
        "- 预测步长/窗口（horizon）：\n"
        "- 关键特征（尤其 OFI/LOB/交易特征；列出原文术语）：\n"
        "- 模型与训练（模型族/损失/训练方式/在线或离线）：\n"
        "- 评价指标（AUC/Accuracy/MAE/RMSE/收益等）：\n"
        "- 主要结论（只写可证据支撑的，逐条列点）：\n"
        "- 局限与适用条件（只写可证据支撑的）：\n"
        "- 与本论文题目“OFI + 美股指数/代表性个股 + 短期预测”的关联（用证据编号支撑）：\n\n"
        "## 2）可追溯证据条目（Evidence Items）\n"
        "至少 8 条，按 E1/E2... 编号。每条必须包含：\n"
        "- 证据类型：定义/方法/实验/结果/局限（五选一）\n"
        "- 定位信息：章节标题/小节标题/表格名/公式附近文本/图名/段落首句片段（不要页码）\n"
        "- 原文关键句：“……”（逐字短引，连续片段，可搜索到，建议 15–40 字）\n"
        "- 我的转述：\n"
        "- 证据等级：A/B/C\n\n"
        "## 3）主题笔记（Topic Notes）\n"
        "3–8 个主题小标题（###）。每段总结必须引用证据编号（如：依据证据 E3、E7）。\n\n"
        "## 4）可直接写进论文的句子草稿（可选）\n"
        "3–6 句，每句末尾标注证据编号（如：依据证据 E2、E5）。\n"
        "-----END_NOTE-----\n"
    )

    user = (
        f"请基于下面论文原文（Markdown）为 `{paper_id}` 生成一篇 `notes_mineru` 文献笔记。\n\n"
        "请严格按下方【noterules.md】中的结构与硬约束输出（包含：大纲映射字段、Evidence Items、数字/对比/因果条款等）。\n"
        "若原文缺失某字段：写【未核验】；不要猜测。\n\n"
        "你必须把最终笔记正文放在标记之间输出，标记外不要输出任何内容（包括思考过程/解释）：\n"
        "- `-----BEGIN_NOTE-----`\n"
        "- `-----END_NOTE-----`\n\n"
        "【必须严格按此模板输出（标题/字段不可改动；只填内容）】\n"
        "-----BEGIN_TEMPLATE-----\n"
        + template
        + "\n-----END_TEMPLATE-----\n\n"
        "关键硬约束提醒（用于防编造）：\n"
        "- 任何数字/提升比例/对比结论/因果归因：除非能给出包含该数字/对比/因果的逐字短引，否则不要写；必要时写【未核验】。\n"
        "- Evidence Items 的“原文关键句”必须逐字来自原文的连续片段，且可搜索到。\n\n"
        "【noterules.md】\n"
        "-----BEGIN_NOTERULES-----\n"
        f"{noterules}\n"
        "-----END_NOTERULES-----\n\n"
        "原文：\n-----BEGIN_REF_MD-----\n"
        f"{source_md}\n"
        "-----END_REF_MD-----\n"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _validate_note_md(paper_id: str, note_md: str) -> tuple[bool, str]:
    """
    最小化结构校验（不改写内容）：
    - 必须含 4 个章节标题
    - Evidence Items 必须至少 8 条（以 '### E' 计数）
    - 建议以 '# <paper_id>' 开头（否则视为失败）
    """
    s = (note_md or "").strip()
    if not s:
        return False, "empty_output"
    # 允许标题不完全等于 paper_id（有些模型会输出英文标题），但必须以一级标题开头
    if not s.startswith("# "):
        return False, "missing_title_header"

    required = [
        "## 1）元数据卡（Metadata Card）",
        "## 2）可追溯证据条目（Evidence Items）",
        "## 3）主题笔记（Topic Notes）",
    ]
    for h in required:
        if h not in s:
            return False, f"missing_section: {h}"
    # 允许两种标题写法（模型偶尔会省略“（可选）”）
    if ("## 4）可直接写进论文的句子草稿（可选）" not in s) and ("## 4）可直接写进论文的句子草稿" not in s):
        return False, "missing_section: ## 4）可直接写进论文的句子草稿（可选）"

    # 允许两种常见写法：
    # 1) "### E1"
    # 2) "- **E1**：" 或 "- E1:"
    e_cnt = len(re.findall(r"^###\s+E\d+\b", s, flags=re.M))
    if e_cnt < 8:
        e_cnt = len(re.findall(r"^\s*[-*]\s*(?:\*\*)?E\d+(?:\*\*)?\b", s, flags=re.M))
    if e_cnt < 8:
        return False, f"insufficient_evidence_items: {e_cnt}"
    return True, ""


def _mk_fix_messages(paper_id: str, source_md: str, noterules: str, bad_output: str) -> list[dict]:
    """
    二次纠错：把上一次不合规输出作为反例，让模型严格按模板重写。
    """
    system = (
        "你是一个严谨的“论文写作与文献笔记助手”。\n"
        "你必须严格按用户给定的模板输出；不要输出思考过程；不要输出任何额外章节。\n"
        "输出必须是 Markdown，不要输出 JSON，不要输出代码块。\n"
        "你的输出会被程序做严格校验：必须以第一行 `# <paper_id>` 开始，且必须包含四个固定二级标题与至少8条 Evidence Items。\n"
    )
    template = (
        "-----BEGIN_NOTE-----\n"
        f"# {paper_id}\n\n"
        "## 1）元数据卡（Metadata Card）\n"
        "- 标题：\n"
        "- 作者：\n"
        "- 年份：\n"
        "- 期刊/会议：\n"
        "- DOI/URL：\n"
        "- 适配章节（映射到论文大纲，写 1–3 个）：\n"
        "- 一句话可用结论（必须含证据编号）：\n"
        "- 可复用证据（列出最关键 3–5 条 E 编号）：\n"
        "- 市场/资产（指数/个股/期货/加密等）：\n"
        "- 数据来源（交易所/数据库/公开数据集名称）：\n"
        "- 频率（tick/quote/trade/分钟/日等）：\n"
        "- 预测目标（方向/收益/价格变化/波动/冲击等）：\n"
        "- 预测视角（点预测/区间/分类/回归）：\n"
        "- 预测步长/窗口（horizon）：\n"
        "- 关键特征（尤其 OFI/LOB/交易特征；列出原文术语）：\n"
        "- 模型与训练（模型族/损失/训练方式/在线或离线）：\n"
        "- 评价指标（AUC/Accuracy/MAE/RMSE/收益等）：\n"
        "- 主要结论（只写可证据支撑的，逐条列点）：\n"
        "- 局限与适用条件（只写可证据支撑的）：\n"
        "- 与本论文题目“OFI + 美股指数/代表性个股 + 短期预测”的关联（用证据编号支撑）：\n\n"
        "## 2）可追溯证据条目（Evidence Items）\n"
        "至少 8 条，按 E1/E2... 编号。每条必须包含：\n"
        "- 证据类型：定义/方法/实验/结果/局限（五选一）\n"
        "- 定位信息：章节标题/小节标题/表格名/公式附近文本/图名/段落首句片段（不要页码）\n"
        "- 原文关键句：“……”（逐字短引，连续片段，可搜索到，建议 15–40 字）\n"
        "- 我的转述：\n"
        "- 证据等级：A/B/C\n\n"
        "## 3）主题笔记（Topic Notes）\n"
        "3–8 个主题小标题（###）。每段总结必须引用证据编号（如：依据证据 E3、E7）。\n\n"
        "## 4）可直接写进论文的句子草稿（可选）\n"
        "3–6 句，每句末尾标注证据编号（如：依据证据 E2、E5）。\n"
        "-----END_NOTE-----\n"
    )
    user = (
        "你上一次的输出不符合结构要求（例如缺少四段式结构/缺少 Evidence Items/输出了思考过程等）。\n"
        "请完全忽略上一次输出，严格按模板重写。\n"
        "硬性要求：最终只输出被标记包裹的笔记正文，标记外不要输出任何内容（包括思考过程/解释）。\n"
        f"笔记正文第一行必须是 `# {paper_id}`。\n\n"
        "【必须严格按此模板输出（标题/字段不可改动；只填内容）】\n"
        "-----BEGIN_TEMPLATE-----\n"
        + template
        + "\n-----END_TEMPLATE-----\n\n"
        "补充提醒：必须遵守 noterules.md 的硬约束（零编造/短引可搜索/数字-短引绑定/对比-短引绑定/因果-短引绑定）。\n\n"
        "【上一次不合规输出（仅供你避免重复错误）】\n"
        "-----BEGIN_BAD_OUTPUT-----\n"
        + (bad_output or "")
        + "\n-----END_BAD_OUTPUT-----\n\n"
        "【原文】\n"
        "-----BEGIN_REF_MD-----\n"
        + source_md
        + "\n-----END_REF_MD-----\n"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _mk_fix_messages_with_reason(
    paper_id: str, source_md: str, noterules: str, bad_output: str, reason: str
) -> list[dict]:
    """
    三次纠错：带上失败原因（缺哪个 section / 证据不足等），要求模型“严格补齐”。
    """
    system = (
        "你是一个严谨的“论文写作与文献笔记助手”。\n"
        "你必须严格按用户给定的模板输出；不要输出思考过程；不要输出任何额外章节。\n"
        "输出必须是 Markdown，不要输出 JSON，不要输出代码块。\n"
        "注意：你的输出会被程序做严格校验，缺任何章节或证据条目都会被判定失败。\n"
    )
    user = (
        f"你上一次的输出不合规，失败原因：{reason}\n"
        "请严格按 noterules.md 的结构输出四个部分，并确保 Evidence Items 至少 8 条。\n"
        "禁止输出任何“解释/思考过程/方法论”，只输出最终笔记正文。\n\n"
        f"硬性要求：输出第一行必须是 `# {paper_id}`。\n"
        "并且必须包含下面四个二级标题（标题文字必须一致）：\n"
        "- ## 1）元数据卡（Metadata Card）\n"
        "- ## 2）可追溯证据条目（Evidence Items）\n"
        "- ## 3）主题笔记（Topic Notes）\n"
        "- ## 4）可直接写进论文的句子草稿（可选）\n\n"
        "补充提醒：所有数字/对比/因果必须有逐字短引支撑，否则写【未核验】或不写。\n\n"
        "【noterules.md】\n"
        "-----BEGIN_NOTERULES-----\n"
        + (noterules or "")
        + "\n-----END_NOTERULES-----\n\n"
        "【上一次不合规输出（仅供你避免重复错误）】\n"
        "-----BEGIN_BAD_OUTPUT-----\n"
        + (bad_output or "")
        + "\n-----END_BAD_OUTPUT-----\n\n"
        "【原文】\n"
        "-----BEGIN_REF_MD-----\n"
        + source_md
        + "\n-----END_REF_MD-----\n"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]
def main() -> int:
    _setup_utf8_stdio()

    ap = argparse.ArgumentParser()
    ap.add_argument("--env-file", default=".apikey.env", help="读取该 env 文件（默认：.apikey.env，位于项目根目录）")
    ap.add_argument("--model", default="", help="模型名（不填则从环境变量 MODEL/OPENAI_MODEL 读取）")
    ap.add_argument("--base-url", default="", help="Base URL（不填则从环境变量 URL/OPENAI_BASE_URL/OPENAI_URL 读取）")
    ap.add_argument("--api-key", default="", help="API Key（不填则从环境变量 API_KEY/OPENAI_API_KEY 读取）")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--only", default="", help="只处理某个 md 文件名（精确匹配），例如：xxx.md")
    ap.add_argument(
        "--only-grep",
        default="",
        help="只处理正文中包含该关键词的 md（简单子串匹配；便于在命令行不方便输入中文文件名时使用）",
    )
    ap.add_argument("--timeout", type=int, default=180, help="单次请求超时秒数（默认 180）")
    args = ap.parse_args()

    # 读取 .env 作为默认值，但不覆盖用户已经在环境变量中显式设置的值
    # 这样用户在系统环境变量里改 MODEL/URL/API_KEY 会立即生效
    _load_env_file((REPO_ROOT / args.env_file).resolve(), override=False)

    # 兼容火山方舟示例命名（不强制要求用户使用）
    if (os.environ.get("API_KEY") or "").strip() and not (os.environ.get("ARK_API_KEY") or "").strip():
        os.environ["ARK_API_KEY"] = (os.environ.get("API_KEY") or "").strip()

    model = (args.model or "").strip() or _env_first("OPENAI_MODEL", "MODEL")
    base_url = (args.base_url or "").strip() or _env_first("OPENAI_BASE_URL", "OPENAI_URL", "URL")
    api_key = (args.api_key or "").strip() or _env_first("OPENAI_API_KEY", "API_KEY", "ARK_API_KEY")

    if not base_url:
        print("未设置 URL（base_url）。请在 .apikey.env 中填写 URL=... 或设置环境变量 URL。")
        return 2
    if not api_key:
        print("未设置 API_KEY。请在 .apikey.env 中填写 API_KEY=... 或设置环境变量 API_KEY。")
        return 2
    if not model:
        print("未设置 MODEL。请在 .apikey.env 中填写 MODEL=... 或设置环境变量 MODEL。")
        return 2

    cfg = OpenAIConfig(api_key=api_key, model=model, base_url=base_url, timeout_s=args.timeout)

    ref_dir = (REPO_ROOT / "ref_mineru").resolve()
    out_dir = (REPO_ROOT / "notes_mineru").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = (out_dir / "_raw").resolve()
    raw_dir.mkdir(parents=True, exist_ok=True)
    progress_path = out_dir / "_progress.jsonl"

    md_files = sorted([p for p in ref_dir.glob("*.md") if p.is_file()])
    if args.only:
        md_files = [p for p in md_files if p.name == args.only]
    if args.only_grep:
        needle = args.only_grep
        picked: list[Path] = []
        for p in md_files:
            try:
                txt = _read_text(p)
            except Exception:
                continue
            if needle in txt:
                picked.append(p)
        md_files = picked

    print(f"[llm_notes_batch] total_md = {len(md_files)}")
    ok = 0
    skipped = 0
    failed = 0

    for md_path in md_files:
        paper_id = md_path.stem
        note_path = out_dir / f"{paper_id}.md"
        raw_path = raw_dir / f"{_safe_filename(paper_id)}.txt"

        if args.resume and note_path.exists() and not args.overwrite:
            skipped += 1
            continue
        if note_path.exists() and not args.overwrite:
            skipped += 1
            continue

        try:
            src_md = _read_text(md_path)
            messages = _mk_messages(paper_id, src_md)
            raw_resp = _retry(lambda: _llm_generate(cfg, messages))
            raw_path.write_text(raw_resp or "", encoding="utf-8", newline="\n")

            note_md = _extract_note_block(paper_id, raw_resp or "")
            ok_struct, reason = _validate_note_md(paper_id, note_md)
            if not ok_struct:
                # 二次纠错：再次请求严格模板输出（仍然只由大模型生成）
                noterules = _load_noterules()
                fix_msgs = _mk_fix_messages(paper_id, src_md, noterules, note_md[:4000])
                raw2 = _retry(lambda: _llm_generate(cfg, fix_msgs))
                # 把二次返回也落 raw（便于排查）
                raw_path.write_text((raw_path.read_text(encoding="utf-8", errors="replace") + "\n\n" + (raw2 or "")).strip() + "\n", encoding="utf-8", newline="\n")
                note_md = _extract_note_block(paper_id, raw2 or "")
                ok_struct, reason = _validate_note_md(paper_id, note_md)
                if not ok_struct:
                    # 三次纠错：带失败原因再试一次
                    fix_msgs2 = _mk_fix_messages_with_reason(
                        paper_id, src_md, noterules, note_md[:4000], reason
                    )
                    raw3 = _retry(lambda: _llm_generate(cfg, fix_msgs2))
                    raw_path.write_text(
                        (raw_path.read_text(encoding="utf-8", errors="replace") + "\n\n" + (raw3 or "")).strip()
                        + "\n",
                        encoding="utf-8",
                        newline="\n",
                    )
                    note_md = _extract_note_block(paper_id, raw3 or "")
                    ok_struct, reason = _validate_note_md(paper_id, note_md)
                    if not ok_struct:
                        raise RuntimeError(f"笔记结构不符合 noterules 模板：{reason}")

            # 仅做结构校验，不改写内容；通过则原样落盘
            note_path.write_text(note_md + "\n", encoding="utf-8", newline="\n")
            ok += 1
            _append_jsonl(
                progress_path,
                {
                    "ts": _now_iso(),
                    "paper_id": paper_id,
                    "status": "ok",
                    "note": note_path.name,
                    "raw": str(raw_path.relative_to(out_dir)).replace("\\", "/"),
                    "model": cfg.model,
                    "base_url": cfg.base_url,
                    "error": "",
                },
            )
        except Exception as e:
            failed += 1
            msg = f"{type(e).__name__}: {e}"
            print(f"[failed] {paper_id}: {msg}")
            _append_jsonl(
                progress_path,
                {
                    "ts": _now_iso(),
                    "paper_id": paper_id,
                    "status": "failed",
                    "note": note_path.name,
                    "raw": str(raw_path.relative_to(out_dir)).replace("\\", "/"),
                    "model": cfg.model,
                    "base_url": cfg.base_url,
                    "error": _one_line(msg),
                },
            )

        time.sleep(0.5)

    print("[llm_notes_batch] done")
    print(f"- ok: {ok}")
    print(f"- skipped: {skipped}")
    print(f"- failed: {failed}")
    print(f"- out: {out_dir}")
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())


