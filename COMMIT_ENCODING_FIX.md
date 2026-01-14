# Git 提交信息编码修复说明

## 问题描述

由于 Windows PowerShell 默认使用 GBK 编码，而 Git/GitHub 使用 UTF-8 编码，导致包含中文的提交信息在 GitHub 上显示为乱码。

## 已修复的配置

以下 Git 配置已设置为 UTF-8：

```bash
git config --global core.quotepath false
git config --global i18n.commitencoding utf-8
git config --global i18n.logoutputencoding utf-8
```

PowerShell 配置文件也已更新，自动设置 UTF-8 编码。

## 历史提交说明

**注意**: 2026-01-15 之前的提交信息可能包含乱码，这是编码问题导致的。实际提交内容（代码文件）不受影响。

### 乱码提交信息对照表

| 乱码显示 | 实际含义 |
|---------|---------|
| `feat: 娣诲姞experiments...` | `feat: 添加experiments实验脚本目录和ML模型完整回测支持` |
| `feat: data model process - 瀹屾垚...` | `feat: data model process - 完成数据处理全流程脚本` |
| `chapter4 complete: TODO...` | `chapter4 complete: TODO占位符规范化 + 实验任务清单` |

## 未来提交建议

### 方法1：使用英文提交信息（推荐）

```bash
git commit -m "feat: add new feature"
git commit -m "fix: fix bug in module X"
```

### 方法2：使用 VS Code 编辑器（支持中文）

```bash
git commit
# 会在 VS Code 中打开，输入中文提交信息并保存
```

### 方法3：使用文件提交（支持中文）

```bash
# 在文本编辑器中写好提交信息（UTF-8编码），保存为 commit_msg.txt
git commit -F commit_msg.txt
```

## 验证编码设置

运行以下命令验证编码配置：

```bash
git config --global --get i18n.commitencoding
# 应该输出: utf-8
```

## 参考

- [Git 中文编码问题解决方案](https://git-scm.com/book/zh/v2/自定义-Git-配置-Git)
- [PowerShell UTF-8 编码设置](https://docs.microsoft.com/powershell/module/microsoft.powershell.core/about/about_character_encoding)
