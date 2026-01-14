# 修复Git提交信息中的乱码
# 使用 git rebase -i 来修改提交信息

# 设置UTF-8编码
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8
chcp 65001 | Out-Null

Write-Host "修复Git提交信息中的乱码..." -ForegroundColor Green

# 提交信息映射（乱码 -> 正确）
$commitFixes = @{
    "feat: 娣诲姞experiments瀹為獙鑴氭湰鐩綍鍜孧L妯″瀷瀹屾暣鍥炴祴鏀寔" = "feat: 添加experiments实验脚本目录和ML模型完整回测支持"
    "feat: data model process - 瀹屾垚鏁版嵁澶勭悊鍏ㄦ祦绋嬭剼鏈?05-16)" = "feat: data model process - 完成数据处理全流程脚本(05-16)"
    "todo_list update: 娣诲姞chapter1-4 TODO娓呭崟 + 缁熶竴缂栧彿鏍煎紡" = "todo_list update: 添加chapter1-4 TODO清单 + 统一编号格式"
    "chapter5 complete: 鐮旂┒缁撹涓庡睍鏈?+ Cont寮曠敤骞翠唤淇敼" = "chapter5 complete: 研究结论与展望 + Cont引用年份修正"
    "chapter4 complete: TODO鍗犱綅绗﹁鑼冨寲 + 瀹為獙浠诲姟娓呭崟" = "chapter4 complete: TODO占位符规范化 + 实验任务清单"
}

Write-Host "`n注意: 此脚本需要手动使用 git rebase -i 来修改提交信息" -ForegroundColor Yellow
Write-Host "或者使用 git commit --amend 来修改最近的提交" -ForegroundColor Yellow

Write-Host "`n建议的修复方法:" -ForegroundColor Cyan
Write-Host "1. 对于最近的提交: git commit --amend -m '正确的提交信息'" -ForegroundColor White
Write-Host "2. 对于历史提交: git rebase -i HEAD~N (N为要修改的提交数量)" -ForegroundColor White
Write-Host "3. 将 'pick' 改为 'reword'，然后修改提交信息" -ForegroundColor White
