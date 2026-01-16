# PostgreSQL + TimescaleDB 安装指南

> 📅 创建时间：2026-01-13
> 🎯 目的：为OFI论文实验搭建时间序列数据库

---

## 📋 安装步骤

### Step 1: 安装 PostgreSQL

#### Windows 安装

1. 下载 PostgreSQL 安装包：
   - 官网：https://www.postgresql.org/download/windows/
   - 推荐版本：**PostgreSQL 16**

2. 运行安装程序，记住以下信息：
   - 安装路径（默认即可）
   - **超级用户密码**（记住这个！）
   - 端口号（默认 5432）

3. 安装完成后，将 PostgreSQL 添加到环境变量：
   ```
   C:\Program Files\PostgreSQL\16\bin
   ```

4. 验证安装：
   ```powershell
   psql --version
   ```

---

### Step 2: 安装 TimescaleDB

#### Windows 安装

1. 下载 TimescaleDB 安装包：
   - 官网：https://docs.timescale.com/self-hosted/latest/install/installation-windows/
   - 选择与你 PostgreSQL 版本匹配的 TimescaleDB

2. 运行安装程序

3. 配置 PostgreSQL 加载 TimescaleDB：
   
   编辑 `postgresql.conf` 文件（通常在 `C:\Program Files\PostgreSQL\16\data\`）：
   ```
   shared_preload_libraries = 'timescaledb'
   ```

4. 重启 PostgreSQL 服务：
   ```powershell
   # 以管理员身份运行
   net stop postgresql-x64-16
   net start postgresql-x64-16
   ```

---

### Step 3: 创建数据库

1. 打开命令行，连接 PostgreSQL：
   ```powershell
   psql -U postgres
   ```
   输入安装时设置的密码

2. 创建数据库和用户：
   ```sql
   -- 创建数据库
   CREATE DATABASE futu_ofi;
   
   -- 创建专用用户（可选）
   CREATE USER ofi_user WITH PASSWORD 'your_password';
   GRANT ALL PRIVILEGES ON DATABASE futu_ofi TO ofi_user;
   
   -- 退出
   \q
   ```

3. 连接到新数据库并启用 TimescaleDB：
   ```powershell
   psql -U postgres -d futu_ofi
   ```
   
   ```sql
   -- 启用 TimescaleDB 扩展
   CREATE EXTENSION IF NOT EXISTS timescaledb;
   
   -- 验证安装
   \dx
   ```
   
   应该能看到 `timescaledb` 在扩展列表中

---

### Step 4: 执行建表脚本

```powershell
# 执行建表SQL
psql -U postgres -d futu_ofi -f "D:\paper project\database\02_create_tables.sql"
```

或者在 psql 中：
```sql
\i 'D:/paper project/database/02_create_tables.sql'
```

---

## 🔧 常用命令

### PostgreSQL 服务管理（管理员权限）

```powershell
# 启动服务
net start postgresql-x64-16

# 停止服务
net stop postgresql-x64-16

# 重启服务
net stop postgresql-x64-16 && net start postgresql-x64-16
```

### psql 常用命令

```sql
-- 列出所有数据库
\l

-- 连接到数据库
\c futu_ofi

-- 列出所有表
\dt

-- 查看表结构
\d orderbook

-- 查看 TimescaleDB 超表
SELECT * FROM timescaledb_information.hypertables;

-- 退出
\q
```

---

## 🐍 Python 连接配置

安装依赖：
```powershell
pip install psycopg2-binary sqlalchemy pandas
```

连接字符串：
```python
# 方式1：psycopg2
import psycopg2
conn = psycopg2.connect(
    host="localhost",
    port=5432,
    database="futu_ofi",
    user="postgres",
    password="your_password"
)

# 方式2：SQLAlchemy
from sqlalchemy import create_engine
engine = create_engine("postgresql://postgres:your_password@localhost:5432/futu_ofi")
```

---

## ❓ 常见问题

### Q1: TimescaleDB 扩展创建失败
- 检查 `postgresql.conf` 中是否添加了 `shared_preload_libraries = 'timescaledb'`
- 确保重启了 PostgreSQL 服务

### Q2: 连接被拒绝
- 检查 PostgreSQL 服务是否启动
- 检查端口号是否正确（默认5432）
- 检查防火墙设置

### Q3: 权限不足
- 使用 postgres 超级用户执行建表脚本
- 或者给 ofi_user 授予足够权限

---

## ✅ 安装检查清单

- [ ] PostgreSQL 安装完成
- [ ] 能执行 `psql --version`
- [ ] TimescaleDB 安装完成
- [ ] 创建了 `futu_ofi` 数据库
- [ ] 启用了 TimescaleDB 扩展
- [ ] 执行了建表脚本
- [ ] Python 能连接数据库
