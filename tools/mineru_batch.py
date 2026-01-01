from __future__ import annotations

import argparse
import ctypes
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _now_iso() -> str:
    # 本地时区 ISO 格式（不强依赖 tz 库）
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _safe_filename(name: str) -> str:
    """
    让文件名在 Windows 下安全（保留中文/英文/数字/常见符号）。
    """
    name = name.strip().replace("\u0000", "")
    name = re.sub(r"[<>:\"/\\\\|?*]+", "_", name)
    name = re.sub(r"\s+", " ", name).strip()
    # Windows 结尾不能是点/空格
    name = name.rstrip(" .")
    return name or "untitled"


def _win_short_path(p: Path) -> str:
    """
    Windows 下 MinerU 对含中文的绝对路径在某些环境中不稳定，这里尽量转 8.3 short path。
    失败则回退原路径。
    """
    if sys.platform != "win32":
        return str(p)
    try:
        GetShortPathNameW = ctypes.windll.kernel32.GetShortPathNameW
        GetShortPathNameW.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_uint32]
        GetShortPathNameW.restype = ctypes.c_uint32
        buf = ctypes.create_unicode_buffer(32768)
        rv = GetShortPathNameW(str(p), buf, len(buf))
        return buf.value if rv else str(p)
    except Exception:
        return str(p)


def _default_source_dirs() -> list[Path]:
    """
    默认优先使用项目内目录（更可复现，避免迁移后仍扫桌面旧路径）：
    - <repo>/开题报告文献
    - <repo>/英文文献期刊

    若项目内目录不存在，再回退到桌面 “论文/…”。
    """
    dirs: list[Path] = []

    # 1) prefer: repo 内部
    for p in (REPO_ROOT / "开题报告文献", REPO_ROOT / "英文文献期刊"):
        if p.exists():
            dirs.append(p)
    if dirs:
        return dirs

    # 2) fallback: Desktop/论文/...
    desktop = Path.home() / "Desktop"
    for p in (desktop / "论文" / "开题报告文献", desktop / "论文" / "英文文献期刊"):
        if p.exists():
            dirs.append(p)
    return dirs


def _iter_pdfs(dirs: list[Path]) -> list[Path]:
    pdfs: list[Path] = []
    for d in dirs:
        if not d.exists():
            continue
        pdfs.extend(sorted(d.rglob("*.pdf")))
    return pdfs


def _read_list_file(p: Path) -> list[Path]:
    lines = p.read_text(encoding="utf-8", errors="ignore").splitlines()
    out: list[Path] = []
    for ln in lines:
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        out.append(Path(s))
    return out


def _find_mineru_cmd() -> str:
    # 优先 PATH
    cmd = "mineru"
    if sys.platform == "win32":
        scripts_dir = Path(sys.executable).resolve().parent / "Scripts"
        exe = scripts_dir / "mineru.exe"
        if exe.exists():
            cmd = str(exe)
    return cmd


def _tail(s: str, n: int = 5000) -> str:
    if not s:
        return ""
    return s[-n:]


def _run_mineru(
    pdf_path: Path,
    raw_out_dir: Path,
    backend: str,
    method: str,
    lang: str,
    device: str,
    start_page: int | None,
    end_page: int | None,
    source: str,
    timeout_sec: int,
    dry_run: bool,
) -> tuple[bool, str]:
    """
    调用 mineru CLI，返回 (ok, message)。
    """
    mineru_cmd = _find_mineru_cmd()
    pdf_arg = _win_short_path(pdf_path.resolve())
    out_arg = _win_short_path(raw_out_dir.resolve())

    cmd: list[str] = [mineru_cmd, "-p", pdf_arg, "-o", out_arg]
    if backend:
        cmd += ["-b", backend]
    if method:
        cmd += ["-m", method]
    if lang:
        cmd += ["-l", lang]
    if device:
        cmd += ["-d", device]
    if start_page is not None:
        cmd += ["-s", str(start_page)]
    if end_page is not None:
        cmd += ["-e", str(end_page)]
    if source:
        cmd += ["--source", source]

    if dry_run:
        return True, "[dry-run] " + " ".join(cmd)

    raw_out_dir.mkdir(parents=True, exist_ok=True)
    try:
        cp = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
            timeout=timeout_sec,
        )
    except FileNotFoundError:
        return False, "未找到命令 `mineru`（或 mineru.exe）。请先安装 MinerU。"
    except subprocess.TimeoutExpired:
        return False, f"MinerU 运行超时（{timeout_sec}s）。"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"

    stderr = (cp.stderr or "").strip()
    stdout = (cp.stdout or "").strip()
    if cp.returncode != 0:
        return False, _tail(stderr or stdout)
    # 部分环境 mineru 可能 stderr 打 Traceback 但 returncode=0
    if "Traceback (most recent call last)" in stderr or "Exception" in stderr or "AttributeError" in stderr:
        return False, _tail(stderr or stdout)
    return True, _tail(stdout or stderr)


def _pick_best_md(raw_out_dir: Path) -> Path | None:
    mds = [p for p in raw_out_dir.rglob("*.md") if p.is_file()]
    if not mds:
        return None
    mds.sort(key=lambda p: p.stat().st_size, reverse=True)
    return mds[0]


@dataclass
class Result:
    status: str  # ok / skipped / failed / failed_no_md
    pdf: str
    md: str | None
    message: str


def main() -> int:
    ap = argparse.ArgumentParser(description="Batch convert PDFs to Markdown via MinerU (accuracy-first).")
    ap.add_argument("--src", action="append", default=None, help="source directory (repeatable)")
    ap.add_argument("--list-file", default=None, help="UTF-8 txt, one pdf path per line")
    ap.add_argument("--out", default="ref_mineru", help="output md dir (relative to repo)")
    ap.add_argument("--backend", default="pipeline", help="mineru backend, e.g. pipeline")
    ap.add_argument("--method", default="auto", help="mineru method")
    ap.add_argument("--lang", default="", help="ch/en/auto (empty = infer)")
    ap.add_argument("--device", default="cpu", help="cpu/cuda")
    ap.add_argument("--source", default="huggingface", help="huggingface/local")
    ap.add_argument("--start-page", type=int, default=None)
    ap.add_argument("--end-page", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--raw", choices=["keep", "delete"], default="delete", help="keep/delete mineru raw outputs")
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    out_dir = (REPO_ROOT / args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    run_log = out_dir / "run.log"
    idx_log = out_dir / "index.jsonl"

    if args.list_file:
        pdfs = _read_list_file(Path(args.list_file))
    else:
        src_dirs = [Path(p) for p in (args.src or [])] if args.src else _default_source_dirs()
        pdfs = _iter_pdfs(src_dirs)

    def _infer_lang(p: Path) -> str:
        if args.lang and args.lang != "auto":
            return args.lang
        stem = p.stem
        return "ch" if re.search(r"[\u4e00-\u9fff]", stem) else "en"

    # 实时写日志 + 实时输出（便于观察进度）
    ok_count = 0
    skip_count = 0
    fail_count = 0
    fail_no_md_count = 0

    # 行缓冲，确保日志实时刷新
    with run_log.open("a", encoding="utf-8", buffering=1) as f_run, idx_log.open(
        "a", encoding="utf-8", buffering=1
    ) as f_idx:
        for pdf in pdfs:
            pdf = pdf.resolve()
            paper_id = _safe_filename(pdf.stem)
            md_path = out_dir / f"{paper_id}.md"

            print(f"processing: {pdf}")

            if md_path.exists() and not args.overwrite:
                r = Result("skipped", str(pdf), str(md_path), "target md exists")
                skip_count += 1
                line = f"[{_now_iso()}] {r.status} {r.pdf} -> {r.md} | {r.message}"
                print("-> skipped")
                f_run.write(line + "\n")
                f_idx.write(json.dumps(r.__dict__, ensure_ascii=False) + "\n")
                continue

            raw_root = out_dir / "_raw"
            raw_out_dir = raw_root / paper_id

            ok, msg = _run_mineru(
                pdf_path=pdf,
                raw_out_dir=raw_out_dir,
                backend=args.backend,
                method=args.method,
                lang=_infer_lang(pdf),
                device=args.device,
                start_page=args.start_page,
                end_page=args.end_page,
                source=args.source,
                timeout_sec=args.timeout,
                dry_run=args.dry_run,
            )
            if not ok:
                r = Result("failed", str(pdf), None, msg)
                fail_count += 1
                line = f"[{_now_iso()}] {r.status} {r.pdf} | {r.message}"
                print("-> failed")
                f_run.write(line + "\n")
                f_idx.write(json.dumps(r.__dict__, ensure_ascii=False) + "\n")
                continue

            best_md = _pick_best_md(raw_out_dir)
            if not best_md:
                r = Result("failed_no_md", str(pdf), None, msg)
                fail_no_md_count += 1
                line = f"[{_now_iso()}] {r.status} {r.pdf} | {r.message}"
                print("-> failed_no_md")
                f_run.write(line + "\n")
                f_idx.write(json.dumps(r.__dict__, ensure_ascii=False) + "\n")
                continue

            if not args.dry_run:
                md_path.write_text(best_md.read_text(encoding="utf-8", errors="ignore"), encoding="utf-8")
                if args.raw == "delete":
                    try:
                        # 只删当前 paper 的 raw 目录，不动 raw_root
                        for sub in sorted(raw_out_dir.rglob("*"), reverse=True):
                            if sub.is_file():
                                sub.unlink(missing_ok=True)
                            else:
                                sub.rmdir()
                        raw_out_dir.rmdir()
                    except Exception:
                        # 清理失败不影响主产物
                        pass

            r = Result("ok", str(pdf), str(md_path), "ok")
            ok_count += 1
            line = f"[{_now_iso()}] {r.status} {r.pdf} -> {r.md} | {r.message}"
            print("-> ok")
            f_run.write(line + "\n")
            f_idx.write(json.dumps(r.__dict__, ensure_ascii=False) + "\n")

        summary = (
            f"[{_now_iso()}] done: ok={ok_count} skipped={skip_count} "
            f"failed={fail_count} failed_no_md={fail_no_md_count}"
        )
        print(summary)
        f_run.write(summary + "\n")

    # 退出码：有失败则 2，否则 0
    return 2 if (fail_count + fail_no_md_count) > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())



