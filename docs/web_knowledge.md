# Agent 搜索知识库 + 百度登录态获取

## 1. 搜索知识库 (test/agent_web_knowledge.py)

流水线: 必应搜索(b_algo) + 百度百科抓取(Edge headless, 自定义 UA 反爬)
→ 信息框 dt/dd 提取 → SQLite 入库 (带来源/URL/时间戳) → 权威真值校验

| 任务 | 结果 |
|------|------|
| 化学元素 | **118/118** (符号/序数 100% 完整, 修正 17 处错位) |
| ISO 639 语言 | **188 条** (代码+中文名) |

生成: elements_encyclopedia.md + language_codes.md

## 2. 百度登录态获取 (test/agent_baidu_behaviors.py + run_baidu_program.py)

机器登录堡垒场景: 复制用户 Firefox profile (cookies.sqlite) → headless Firefox
+ geckodriver → 登录态确认 (账号掩码, 含全角星号正则) → 百科词条 + 贴吧热帖入库

结果: 登录态 195******35 ✓ | 百科 3 词条 + 贴吧 2 吧真实热帖 → 知识库 + 报告

调试要点:
- 百度账号掩码用全角星号 (U+FF0A) — 正则需覆盖 [\*\uFF0A\u2022\u2217]
- raw.githubusercontent 被墙 → GitHub API + codeload zip

## 3. 冯·诺依曼指令化冲浪 (test/von_surf_engine.py)

固定执行器 + BEHAVE/REFLECT/CORRECT 指令 (12-23 条全在 DB):

| 版本 | 指令 | 准确率 |
|------|------|--------|
| 指令化基础 | 12 条 (收集/生成) | 85 |
| +反射纠偏 REFLECT/CORRECT | 18 条 | **86** (噪声 16.7%→0%) |
| +反馈式驱动 derive_queries | 23 条 | **90** (聚类 2→5) |
| +分层记忆按需提取 | 19 条 | 机制验证 (抽象/隔离统计) |

## 结论

搜索/整理解耦协作 (生产者写 L1, 消费者读 L1 写 L2/L3, 经需求信号互驱) 闭环成立;
登录态复用 (profile cookie) 有效绕过登录堡垒。
