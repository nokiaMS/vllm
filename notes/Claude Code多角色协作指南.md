# Claude Code 多角色协作完全指南

Claude Code 提供了多种方式让多个"角色"（Agent）协同完成一个任务，从简单的子代理分工到多实例团队协作，覆盖从个人开发到复杂工程的全场景。

---

## 一、Subagent（子代理）— 最常用

在对话中，Claude 可以派生专门的子代理去处理子任务，每个子代理有独立的上下文窗口和工具权限。

### 1.1 内置子代理类型

| 类型 | 用途 | 工具权限 | 速度 |
|------|------|---------|------|
| **Explore** | 快速搜索和分析代码库 | 只读（Glob/Grep/Read） | 快 |
| **Plan** | 设计实现方案 | 只读 | 中 |
| **general-purpose** | 复杂多步骤任务 | 全部工具 | 慢（功能最全） |

### 1.2 使用方式

直接在对话中告诉 Claude 使用不同的 Agent：

```
"用 Explore agent 找出所有认证相关的文件"
"用 Plan agent 设计一个重构方案"
"并行启动两个 agent，一个搜索 API 路由，另一个搜索数据库查询"
```

Claude 会自动派生子代理，子代理完成后将结果汇报给主 Agent。

### 1.3 自定义子代理

在项目根目录创建 `.claude/agents/` 目录下的 Markdown 文件：

```markdown
# 文件：.claude/agents/code-reviewer.md

---
name: code-reviewer
description: 代码审查专家，分析安全性、性能和代码风格
tools: [Read, Grep, Glob, Bash]
model: sonnet
color: blue
---

你是一个资深代码审查专家，审查代码时重点关注：
- 安全漏洞（SQL 注入、XSS、权限绕过等）
- 性能问题（N+1 查询、内存泄漏、不必要的计算）
- 代码风格和可维护性
- 是否遵循项目现有的设计模式

提供具体的、可操作的反馈，每个问题给出修复建议。
```

```markdown
# 文件：.claude/agents/test-writer.md

---
name: test-writer
description: 测试工程师，编写全面的单元测试和集成测试
tools: [Read, Grep, Glob, Write, Edit, Bash]
model: sonnet
---

你是一个测试工程师，编写测试时遵循以下原则：
- 每个测试只验证一个行为
- 使用 AAA 模式（Arrange-Act-Assert）
- 覆盖正常路径、边界条件和错误场景
- 测试命名清晰表达预期行为
```

然后在对话中使用：

```
"用 code-reviewer 检查 src/auth/ 目录下的所有文件"
"用 test-writer 为 src/payment.py 编写测试"
```

### 1.4 子代理的工作机制

```
你的请求
  │
  ▼
主 Agent（Claude）
  │  分析任务，决定是否需要子代理
  │
  ├─→ 派生 Explore Agent ──→ 搜索代码 ──→ 返回结果
  │                                          │
  ├─→ 派生 Plan Agent ────→ 设计方案 ──→ 返回结果
  │                                          │
  ▼                                          ▼
主 Agent 汇总所有结果，继续执行或回复你
```

关键特性：
- **独立上下文**：子代理有自己的上下文窗口，不会污染主对话
- **并行执行**：多个子代理可以同时运行
- **结果汇总**：子代理的输出返回给主 Agent，由主 Agent 综合处理

---

## 二、Agent Teams（代理团队）— 多实例协作

实验性功能，让多个 Claude Code 实例像一个团队一样协作。

### 2.1 启用

在 `~/.claude/settings.json` 中添加：

```json
{
  "env": {
    "CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS": "1"
  }
}
```

或设置环境变量：

```bash
export CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1
```

### 2.2 架构

```
         Team Lead（团队负责人）
        /       |        \
  Backend    Frontend    DevOps
  Expert     Expert      Expert
  (后端)     (前端)      (运维)
     │          │           │
  独立工作   独立工作    独立工作
     │          │           │
     └───── 通过邮箱系统互相通信 ─────┘
```

| 角色 | 职责 |
|------|------|
| **Team Lead** | 分配任务、协调进度、审查结果、合并输出 |
| **Teammate** | 独立完成分配的子任务，通过邮箱向 Lead 汇报 |

### 2.3 使用示例

```
创建一个代理团队来重构支付模块：
- TeamLead：协调和审查所有变更
- BackendDev：重构后端业务逻辑和 API
- TestDev：编写单元测试和集成测试
- DocDev：更新 API 文档和 README

先让 BackendDev 分析当前代码结构。
```

### 2.4 终端操作

- **切换 Agent 视图**：`Shift+Up` / `Shift+Down`
- **分屏模式**（推荐 3 个以上 Agent 时）：每个 Agent 占一个 tmux/iTerm2 面板
- **查看任务列表**：所有 Agent 共享 Task List，可看到依赖关系

---

## 三、Worktree（工作树）— 隔离并行

Git Worktree 允许多个 Agent 在独立的分支中并行工作，互不冲突，特别适合"同一问题多种方案"的对比场景。

### 3.1 使用方式

在对话中说"在 worktree 中工作"：

```
"在 worktree 中实现 OAuth2 认证方案"
```

或使用 Agent 工具的 `isolation: "worktree"` 参数。

### 3.2 典型场景

**方案对比：**

```
Worktree A: 用 OAuth2 实现认证
Worktree B: 用 JWT 实现认证
Worktree C: 用 SAML 实现认证

→ 三个 Agent 各自独立工作
→ 你对比三个分支的实现
→ 选最优方案合并到主分支
```

**并行开发（不同模块）：**

```
Worktree 1: 重构 backend/ 目录
Worktree 2: 重构 frontend/ 目录
Worktree 3: 更新 tests/ 目录
Worktree 4: 更新配置文件

→ 同时进行，互不冲突
→ 完成后分别合并
```

### 3.3 清理规则

- **无变更**：Worktree 和分支自动删除
- **有变更**：提示你选择保留或删除
- 手动退出：使用 `ExitWorktree` 工具，选择 `keep` 或 `remove`

---

## 四、内置多角色命令

### 4.1 /simplify — 并行代码审查

自动派生 3 个审查 Agent 并行工作：

```
Agent 1：检查代码复用机会
Agent 2：检查代码质量
Agent 3：检查执行效率
```

```bash
# 审查最近的代码变更
/simplify

# 聚焦特定方面
/simplify focus on memory efficiency
```

三个 Agent 的结果汇总后，自动修复发现的问题。

### 4.2 /batch — 批量并行修改

为每个独立的文件/模块派生一个 Agent，在隔离的 Worktree 中并行执行，速度比串行快 10 倍。

```bash
/batch
```

流程：
1. 分析变更策略
2. 识别独立的修改单元（文件/模块）
3. 为每个单元派生一个 Agent（各在独立 Worktree 中）
4. 所有 Agent 并行执行
5. 结果汇总合并

适用场景：
- 全代码库的风格统一
- 批量替换依赖库
- 为未覆盖的模块批量生成测试
- API 版本升级

### 4.3 /loop — 定时循环任务

```bash
# 每 5 分钟运行测试
/loop 5m "run tests"

# 每 10 分钟检查代码规范（默认间隔）
/loop "check for linting errors"

# 每小时安全审查
/loop 1h "/security-review"
```

---

## 五、常用协作模式

### 模式一：研究 → 计划 → 执行

```
Explore Agent（快速只读搜索）
  ↓ "找到 15 个认证相关文件，3 个存在安全隐患"
Plan Agent（设计方案）
  ↓ "建议分 5 步重构，先改核心模块再改依赖"
主 Agent（执行修改）
  ↓ 逐步实现代码变更
test-runner Agent（验证）
  ↓ "所有测试通过"
```

**使用方式：**

```
"先用 Explore agent 找出所有认证相关的文件和已知问题，
 然后用 Plan agent 设计重构方案，
 方案确认后开始执行修改，
 最后运行测试验证。"
```

### 模式二：并行专家搜索

```
你："并行做以下搜索：
     1. 一个 agent 搜索所有 API 路由定义
     2. 另一个 agent 搜索所有数据库查询
     3. 再一个 agent 搜索所有中间件配置"

→ 三个子代理同时工作
→ 结果汇总后，你获得全面的代码地图
```

### 模式三：开发 → 审查 → 修复循环

```
主 Agent 编写代码
  ↓
code-reviewer Agent 审查
  ↓ "发现 3 个安全问题，2 个性能问题"
主 Agent 修复所有问题
  ↓
test-runner Agent 运行测试
  ↓ "全部通过，覆盖率 92%"
```

### 模式四：Worktree 方案竞争

```
Worktree A: Agent 用 Redis 实现缓存
Worktree B: Agent 用内存实现缓存
Worktree C: Agent 用文件系统实现缓存

→ 你对比三个方案的性能和复杂度
→ 选择 Redis 方案
→ 合并到主分支
```

### 模式五：Team Lead 模式

```
         Team Lead
        /    |    \
  后端     前端    测试
  专家     专家    专家
   │        │       │
修改API  改UI组件  写E2E测试
   │        │       │
   └── 通过邮箱协调接口契约 ──┘

→ Lead 审查所有变更
→ 合并为一个完整的 Feature
```

---

## 六、Claude Agent SDK（编程方式）

用 Python 或 TypeScript 构建自定义多 Agent 应用。

### 6.1 安装

```bash
# Python 3.10+
pip install claude-agent-sdk

# Node 18+
npm install @anthropic-ai/sdk
```

### 6.2 多角色协作示例（Python）

```python
from anthropic_agent_sdk import Agent

# 定义不同角色
researcher = Agent(
    instructions="你是代码分析专家，擅长发现问题和改进机会",
    tools=["file_read", "shell_execute"]
)

developer = Agent(
    instructions="你是高级开发者，编写高质量、安全的代码",
    tools=["file_read", "file_write", "shell_execute"]
)

tester = Agent(
    instructions="你是测试专家，编写全面的测试用例",
    tools=["file_read", "file_write", "shell_execute"]
)

# 串行协作流程
async def refactor_module(module_path: str):
    # 阶段 1：分析
    analysis = await researcher.query(f"分析 {module_path} 的代码质量和安全问题")

    # 阶段 2：修复
    fix = await developer.query(f"根据以下分析修复代码：\n{analysis}")

    # 阶段 3：测试
    test = await tester.query(f"为 {module_path} 编写测试，确保修复正确")

    return test
```

### 6.3 自定义工具

```python
from anthropic_agent_sdk import Agent

agent = Agent()

@agent.tool
def run_benchmark(file_path: str, iterations: int = 100) -> str:
    """运行性能基准测试"""
    import subprocess
    result = subprocess.run(
        ["python", "-m", "pytest", file_path, f"--count={iterations}", "--benchmark-only"],
        capture_output=True, text=True
    )
    return result.stdout
```

### 6.4 会话管理（Agent 间传递上下文）

```python
# Agent 1 的输出可以作为 Agent 2 的输入
async for message in agent1.query("找出所有 API 路由"):
    pass
session1_id = message.session_id

# Agent 2 引用 Agent 1 的发现
async for message in agent2.query(
    "基于这些路由实现测试",
    context_from=session1_id
):
    pass
```

---

## 七、快速选择指南

| 需求 | 推荐方式 | 复杂度 |
|------|---------|--------|
| 单次对话中分步协作 | **Subagent** | 低 |
| 并行搜索/分析代码 | **多个 Explore Subagent** | 低 |
| 代码审查 | **`/simplify`** | 低 |
| 批量修改大量文件 | **`/batch`** | 中 |
| 多种方案对比 | **Worktree** | 中 |
| 定时持续检查 | **`/loop`** | 低 |
| 多 Agent 长期协作 | **Agent Teams**（实验功能） | 高 |
| 自建多 Agent 应用 | **Claude Agent SDK** | 高 |

---

## 八、最佳实践

### 8.1 Subagent 使用原则

1. **只在必要时派生子代理** — 简单的单文件操作不需要子代理，直接做即可
2. **用 Explore 做搜索，用 general-purpose 做修改** — Explore 快但只读，不要让它做写操作
3. **并行优于串行** — 多个独立的搜索/分析任务应并行派生，不要一个接一个
4. **给子代理清晰的单一任务** — "搜索所有 API 路由" 比 "分析整个项目" 效果好得多
5. **避免过度嵌套** — 子代理不应再派生子代理，保持一层即可

### 8.2 Agent Teams 使用原则

6. **Team Lead 只协调不执行** — Lead 负责分配、审查、合并，具体工作交给 Teammate
7. **每个 Teammate 职责单一** — "后端专家"不应同时负责写测试和文档
8. **先让一个 Agent 分析，再分配任务** — 避免盲目分工
9. **控制团队规模** — 2-4 个 Agent 最佳，超过 5 个协调成本大于收益

### 8.3 Worktree 使用原则

10. **方案对比时用 Worktree** — 同一问题多种解法，各建一个 Worktree
11. **不同模块并行修改时用 Worktree** — 避免 Git 冲突
12. **及时清理** — 选定方案后删除其他 Worktree，避免分支堆积

### 8.4 性能与成本

13. **Explore Agent 最便宜** — 只读操作，上下文小，优先使用
14. **`--set full` 用 Sonnet 不用 Opus** — 子代理通常 Sonnet 就够用，Opus 留给主 Agent
15. **限制子代理的搜索范围** — "在 src/auth/ 目录下搜索" 比 "在整个项目搜索" 快得多

---

## 九、完整实战案例：重构认证模块

以下是一个完整的多角色协作示例，演示如何用 Claude Code 的多种 Agent 能力重构一个认证模块。

### 场景

项目的认证模块使用了过时的 Session 方案，需要迁移到 JWT，同时保证向后兼容、测试覆盖和文档更新。

### 第一阶段：情报收集（并行 Explore）

```
你："并行帮我做以下调研：
     1. 用 Explore agent 找出所有认证相关的文件和函数
     2. 用另一个 Explore agent 找出所有依赖认证的 API 路由
     3. 用第三个 Explore agent 找出现有的认证测试覆盖情况"
```

三个 Explore Agent 并行工作，各自返回结果：

```
Agent 1 结果：
  - src/auth/session.py（核心认证逻辑）
  - src/auth/middleware.py（中间件）
  - src/auth/utils.py（工具函数）
  - src/models/user.py（用户模型，含 session 字段）

Agent 2 结果：
  - 47 个 API 路由依赖 @login_required 装饰器
  - 12 个路由使用了 session.get("user_id") 直接访问

Agent 3 结果：
  - tests/test_auth.py：仅 23% 覆盖率
  - 缺少中间件测试和边界条件测试
```

### 第二阶段：方案设计（Plan Agent）

```
你："基于以上调研结果，用 Plan agent 设计 JWT 迁移方案"
```

Plan Agent 输出：

```
迁移方案（5 步）：
1. 新建 src/auth/jwt.py，实现 JWT 签发/验证
2. 修改 middleware.py 支持 JWT + Session 双模式（过渡期）
3. 逐步修改 47 个路由，移除 session 直接访问
4. 编写完整测试（目标覆盖率 >90%）
5. 迁移完成后移除 Session 代码
```

你确认方案后进入执行阶段。

### 第三阶段：并行实现（Worktree + 多 Agent）

```
你："在 worktree 中执行步骤 1-2 的代码修改"
```

主 Agent 在 Worktree 中实现核心变更：
- 创建 `src/auth/jwt.py`
- 修改 `src/auth/middleware.py` 支持双模式

同时：

```
你："并行用 test-writer agent 为新的 JWT 模块编写测试"
```

test-writer Agent 同步编写测试文件 `tests/test_jwt.py`。

### 第四阶段：审查（/simplify）

```
/simplify
```

三个审查 Agent 并行检查：
- Agent 1：发现 JWT 密钥硬编码（安全问题）→ 自动修复为环境变量
- Agent 2：发现 token 过期时间未配置化 → 自动提取为配置项
- Agent 3：发现一个不必要的数据库查询 → 自动优化

### 第五阶段：验证

```
你："运行所有测试验证修改"
```

```
测试结果：
  tests/test_jwt.py ................ 18 passed
  tests/test_auth.py ............... 12 passed
  tests/test_middleware.py ......... 8 passed
  tests/test_routes.py ............. 47 passed

  覆盖率：91.3%（目标 >90% ✓）
```

### 全流程角色分工总结

| 阶段 | 角色 | 工具 | 耗时 |
|------|------|------|------|
| 情报收集 | 3 个 Explore Agent（并行） | Glob/Grep/Read | ~30 秒 |
| 方案设计 | 1 个 Plan Agent | Read | ~20 秒 |
| 核心开发 | 主 Agent（Worktree 隔离） | Read/Edit/Write | ~3 分钟 |
| 测试编写 | test-writer Agent（并行） | Read/Write | ~2 分钟 |
| 代码审查 | /simplify（3 个 Agent 并行） | Read/Edit | ~1 分钟 |
| 验证 | 主 Agent | Bash（pytest） | ~30 秒 |
| **总计** | **9 个 Agent 角色** | | **~7 分钟** |

如果纯串行操作，相同任务预计需要 20-30 分钟。多角色并行将效率提升了 3-4 倍。

---

> **参考来源**：
> - [Create custom subagents - Claude Code Docs](https://code.claude.com/docs/en/sub-agents)
> - [Orchestrate teams of Claude Code sessions - Claude Code Docs](https://code.claude.com/docs/en/agent-teams)
> - [Agent SDK overview - Claude API Docs](https://platform.claude.com/docs/en/agent-sdk/overview)
> - [How to Use Claude Code Subagents to Parallelize Development](https://zachwills.net/how-to-use-claude-code-subagents-to-parallelize-development/)
> - [Awesome Claude Code Subagents](https://github.com/VoltAgent/awesome-claude-code-subagents)
> - [Claude Code Worktrees Guide](https://claudefa.st/blog/guide/development/worktree-guide)
> - [Code Agent Orchestra](https://addyosmani.com/blog/code-agent-orchestra/)
