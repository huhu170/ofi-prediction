# ============================================================
# OFI论文 - 数据库快速配置脚本 (PowerShell)
# 前提：已安装 PostgreSQL 并添加到环境变量
# ============================================================

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  OFI论文 - 数据库快速配置" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# 配置（根据实际情况修改）
$PG_USER = "postgres"
$PG_DATABASE = "futu_ofi"
$SCRIPT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path

Write-Host "`n请输入 PostgreSQL 密码:" -ForegroundColor Yellow
$PG_PASSWORD = Read-Host -AsSecureString
$BSTR = [System.Runtime.InteropServices.Marshal]::SecureStringToBSTR($PG_PASSWORD)
$PG_PASSWORD_PLAIN = [System.Runtime.InteropServices.Marshal]::PtrToStringAuto($BSTR)

# 设置环境变量（避免每次输入密码）
$env:PGPASSWORD = $PG_PASSWORD_PLAIN

Write-Host "`n[Step 1] 检查 PostgreSQL 连接..." -ForegroundColor Green
try {
    $version = psql -U $PG_USER -c "SELECT version();" -t 2>$null
    Write-Host "✅ PostgreSQL 连接成功" -ForegroundColor Green
} catch {
    Write-Host "❌ PostgreSQL 连接失败，请检查服务是否启动" -ForegroundColor Red
    exit 1
}

Write-Host "`n[Step 2] 创建数据库 $PG_DATABASE ..." -ForegroundColor Green
psql -U $PG_USER -c "CREATE DATABASE $PG_DATABASE;" 2>$null
if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ 数据库创建成功" -ForegroundColor Green
} else {
    Write-Host "⚠️ 数据库可能已存在，继续..." -ForegroundColor Yellow
}

Write-Host "`n[Step 3] 启用 TimescaleDB 扩展..." -ForegroundColor Green
psql -U $PG_USER -d $PG_DATABASE -c "CREATE EXTENSION IF NOT EXISTS timescaledb;"
if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ TimescaleDB 扩展启用成功" -ForegroundColor Green
} else {
    Write-Host "❌ TimescaleDB 扩展启用失败" -ForegroundColor Red
    Write-Host "   请确认 TimescaleDB 已正确安装" -ForegroundColor Red
    exit 1
}

Write-Host "`n[Step 4] 执行建表脚本..." -ForegroundColor Green
$SQL_FILE = Join-Path $SCRIPT_DIR "02_create_tables.sql"
if (Test-Path $SQL_FILE) {
    psql -U $PG_USER -d $PG_DATABASE -f $SQL_FILE
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ 建表脚本执行成功" -ForegroundColor Green
    } else {
        Write-Host "❌ 建表脚本执行失败" -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "❌ 找不到建表脚本: $SQL_FILE" -ForegroundColor Red
    exit 1
}

Write-Host "`n[Step 5] 验证表结构..." -ForegroundColor Green
$tables = psql -U $PG_USER -d $PG_DATABASE -t -c "SELECT tablename FROM pg_tables WHERE schemaname = 'public';"
Write-Host "已创建的表:" -ForegroundColor Cyan
Write-Host $tables

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "  ✅ 数据库配置完成！" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "`n下一步:" -ForegroundColor Yellow
Write-Host "  1. 修改 03_test_connection.py 中的密码" -ForegroundColor White
Write-Host "  2. 运行: python 03_test_connection.py" -ForegroundColor White
Write-Host "  3. 开始数据采集" -ForegroundColor White

# 清除密码
$env:PGPASSWORD = ""
