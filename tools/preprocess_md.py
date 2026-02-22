"""
preprocess_md.py — 预处理 paper.md，解决 pandoc 转换的两个已知问题：
1. $$...\tag{X}$$ → 去掉 \tag{}，公式后追加右对齐编号文本
2. 参考文献 [n] 开头被解析为链接 → 转义为 \[n\]
输出 paper_preprocessed.md
"""
import re
import sys
from pathlib import Path


def process_equation_tags(text: str) -> str:
    """将 $$...\tag{X}$$ 替换为不含 \tag 的公式 + 编号行"""
    tag_re = re.compile(r"\$\$(.*?)\\tag\{([^}]+)\}\$\$", re.DOTALL)

    def repl(m):
        formula = m.group(1).strip()
        tag_num = m.group(2).strip()
        return f"$${formula}$$\n\n（{tag_num}）\n"

    return tag_re.sub(repl, text)


def escape_ref_numbers(text: str) -> str:
    """将参考文献区域的 [n] 编号转义，并确保条目间有空行"""
    lines = text.split("\n")
    in_ref = False
    result = []
    prev_was_ref = False
    for line in lines:
        if line.strip().startswith("# 参考文献"):
            in_ref = True
        if in_ref and re.match(r"^\[(\d+)\]\s", line):
            if prev_was_ref:
                result.append("")
            line = re.sub(r"^\[(\d+)\]", r"\\[\1\\]", line)
            prev_was_ref = True
        else:
            prev_was_ref = False
        result.append(line)
    return "\n".join(result)


def main():
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("paper.md")
    dst = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("paper_preprocessed.md")

    text = src.read_text(encoding="utf-8")

    print(f"  [1/2] Processing \\tag{{}} in equations...")
    tag_count = len(re.findall(r"\\tag\{", text))
    text = process_equation_tags(text)
    print(f"        Processed {tag_count} equations with \\tag{{}}")

    print(f"  [2/2] Escaping reference [n] numbers...")
    ref_count = len(re.findall(r"^\[\d+\]\s", text, re.MULTILINE))
    text = escape_ref_numbers(text)
    print(f"        Escaped {ref_count} reference entries")

    dst.write_text(text, encoding="utf-8")
    print(f"  Done: {dst}")


if __name__ == "__main__":
    main()
