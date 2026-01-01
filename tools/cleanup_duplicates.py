from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


def _sha1_bytes(b: bytes) -> str:
    return hashlib.sha1(b).hexdigest()


def _norm_name(p: Path) -> str:
    """
    归一化“可能的重复文件名”，处理：
    - foo (1).md
    - foo__来源.md
    """
    stem = p.stem
    stem = stem.replace(" (1)", "").replace("(1)", "")
    if "__" in stem:
        stem = stem.split("__", 1)[0]
    return stem.strip()


def main() -> int:
    ap = argparse.ArgumentParser(description="Cleanup duplicate md files in ref_mineru/")
    ap.add_argument("--dir", default="ref_mineru", help="target directory (relative to repo)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--report", default="ref_mineru/cleanup_duplicates_report.txt")
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[1]
    d = (repo / args.dir).resolve()
    report_path = (repo / args.report).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)

    files = [p for p in d.glob("*.md") if p.is_file()]
    groups: dict[str, list[Path]] = {}
    for p in files:
        groups.setdefault(_norm_name(p), []).append(p)

    lines: list[str] = []
    deleted = 0
    for base, ps in sorted(groups.items(), key=lambda kv: kv[0]):
        if len(ps) <= 1:
            continue
        # 先按“内容 hash + 大小”判断重复
        infos = []
        for p in ps:
            b = p.read_bytes()
            infos.append((p, len(b), _sha1_bytes(b)))
        # 先按大小降序保留最大者
        infos.sort(key=lambda x: x[1], reverse=True)
        keep = infos[0][0]
        lines.append(f"[KEEP] {keep.name}")
        for p, sz, h in infos[1:]:
            if p == keep:
                continue
            lines.append(f"[DEL ] {p.name} (size={sz}, sha1={h})")
            if not args.dry_run:
                p.unlink(missing_ok=True)
            deleted += 1
        lines.append("")

    report_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    print(f"done. deleted={deleted}. report={report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



