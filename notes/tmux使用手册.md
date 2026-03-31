# tmux 使用手册

tmux（Terminal Multiplexer）是一个终端复用器，允许你在一个终端窗口中创建、访问和控制多个终端会话。即使断开连接，会话仍在后台运行。

---

## 核心概念

| 概念 | 说明 |
|------|------|
| **Session（会话）** | 最顶层的容器，可包含多个窗口 |
| **Window（窗口）** | 类似浏览器标签页，一个会话可包含多个窗口 |
| **Pane（窗格）** | 窗口内的分屏区域，一个窗口可分为多个窗格 |

**前缀键**：tmux 的快捷键都需要先按前缀键，默认为 `Ctrl+b`，下文用 `prefix` 表示。

---

## 1. 会话管理（Session）

### 1.1 新建会话

```bash
# 新建一个默认会话（自动编号 0, 1, 2...）
tmux

# 新建一个命名会话
tmux new -s myproject

# 新建会话并指定窗口名称
tmux new -s myproject -n editor
```

### 1.2 查看会话列表

```bash
# 命令行查看
tmux ls
# 等价写法
tmux list-sessions

# 输出示例：
# myproject: 2 windows (created Mon Mar 31 10:00:00 2026)
# dev: 1 windows (created Mon Mar 31 09:30:00 2026)
```

快捷键方式：`prefix` + `s` — 交互式选择会话。

### 1.3 分离会话（Detach）

将当前会话放入后台，终端恢复正常状态。

```bash
# 快捷键（推荐）
prefix + d

# 命令方式
tmux detach  --在session内执行此命令,退出session界面转为后台执行.
```

### 1.4 重新连接会话（Attach）

```bash
# 连接到最近的会话
tmux attach
# 简写
tmux a

# 连接到指定会话
tmux attach -t myproject
# 简写
tmux a -t myproject
```

### 1.5 杀死会话

```bash
# 杀死指定会话
tmux kill-session -t myproject

# 杀死除当前会话外的所有会话
tmux kill-session -a

# 杀死除指定会话外的所有会话
tmux kill-session -a -t myproject
```

### 1.6 重命名会话

```bash
# 命令行方式
tmux rename-session -t old_name new_name

# 快捷键（在会话内）
prefix + $
# 然后输入新名称，按回车确认
```

### 1.7 切换会话

```bash
# 快捷键：交互式选择
prefix + s

# 切换到上一个会话
prefix + (

# 切换到下一个会话
prefix + )

# 命令方式
tmux switch -t myproject
```

---

## 2. 窗口管理（Window）

### 2.1 新建窗口

```bash
# 快捷键
prefix + c

# 命令方式
tmux new-window

# 新建窗口并命名
tmux new-window -n logs
```

### 2.2 切换窗口

```bash
# 切换到下一个窗口
prefix + n

# 切换到上一个窗口
prefix + p

# 切换到指定编号的窗口（0-9）
prefix + 0    # 切到第 0 个窗口
prefix + 1    # 切到第 1 个窗口
prefix + 3    # 切到第 3 个窗口

# 交互式窗口列表
prefix + w
```

### 2.3 重命名窗口

```bash
# 快捷键
prefix + ,
# 然后输入新名称，按回车确认

# 命令方式
tmux rename-window new_name
```

### 2.4 关闭窗口

```bash
# 关闭当前窗口（快捷键）
prefix + &
# 会提示确认，按 y 确认

# 命令方式
tmux kill-window

# 关闭指定窗口
tmux kill-window -t 2
```

### 2.5 移动窗口

```bash
# 交换窗口位置：将窗口 2 和窗口 1 互换
tmux swap-window -s 2 -t 1

# 将当前窗口移到编号 0
tmux move-window -t 0
```

---

## 3. 窗格管理（Pane）

### 3.1 分屏

```bash
# 水平分屏（上下分割）
prefix + "

# 垂直分屏（左右分割）
prefix + %
```

### 3.2 切换窗格

```bash
# 方向键切换
prefix + ↑    # 切到上方窗格
prefix + ↓    # 切到下方窗格
prefix + ←    # 切到左方窗格
prefix + →    # 切到右方窗格

# 顺序切换到下一个窗格
prefix + o

# 切换到上一个活跃窗格
prefix + ;
```

### 3.3 调整窗格大小

```bash
# 按住 prefix，然后按方向键调整
prefix + Ctrl+↑    # 向上扩展
prefix + Ctrl+↓    # 向下扩展
prefix + Ctrl+←    # 向左扩展
prefix + Ctrl+→    # 向右扩展

# 命令方式：精确调整（单位为行/列）
tmux resize-pane -U 5    # 向上扩展 5 行
tmux resize-pane -D 5    # 向下扩展 5 行
tmux resize-pane -L 10   # 向左扩展 10 列
tmux resize-pane -R 10   # 向右扩展 10 列
```

### 3.4 窗格缩放（全屏/还原）

```bash
# 将当前窗格最大化（再按一次还原）
prefix + z
```

### 3.5 关闭窗格

```bash
# 快捷键
prefix + x
# 会提示确认，按 y 确认

# 或者直接在窗格内输入
exit
```

### 3.6 交换窗格位置

```bash
# 与上一个窗格交换
prefix + {

# 与下一个窗格交换
prefix + }

# 命令方式
tmux swap-pane -s 0 -t 1
```

### 3.7 窗格转为独立窗口

```bash
# 将当前窗格拆分为独立窗口
prefix + !
```

### 3.8 切换窗格布局

```bash
# 在预设布局间循环切换
prefix + Space

# 可用布局：even-horizontal, even-vertical, main-horizontal, main-vertical, tiled
# 命令方式指定布局
tmux select-layout even-horizontal
tmux select-layout tiled
```

---

## 4. 复制模式（Copy Mode）

tmux 自带滚动和复制功能，无需鼠标。

### 4.1 进入复制模式

```bash
prefix + [
```

### 4.2 复制模式内操作（vi 风格）

| 按键 | 功能 |
|------|------|
| `q` | 退出复制模式 |
| `h/j/k/l` | 上下左右移动光标 |
| `Ctrl+u` | 向上翻半页 |
| `Ctrl+d` | 向下翻半页 |
| `g` | 跳到顶部 |
| `G` | 跳到底部 |
| `/` | 向下搜索 |
| `?` | 向上搜索 |
| `n` | 下一个搜索结果 |
| `N` | 上一个搜索结果 |
| `Space` | 开始选择 |
| `Enter` | 复制选中内容并退出 |

### 4.3 粘贴

```bash
prefix + ]
```

### 4.4 查看粘贴缓冲区

```bash
tmux list-buffers

# 选择并粘贴指定缓冲区
tmux choose-buffer
# 快捷键
prefix + =
```

---

## 5. 命令模式

```bash
# 进入 tmux 命令行
prefix + :

# 然后可输入任意 tmux 命令，例如：
:new-window -n logs
:split-window -h
:resize-pane -D 10
:setw synchronize-panes on    # 同步输入到所有窗格
:setw synchronize-panes off   # 关闭同步输入
```

---

## 6. 实用技巧

### 6.1 同步输入（向所有窗格发送相同命令）

```bash
# 开启同步
prefix + : 然后输入 setw synchronize-panes on

# 关闭同步
prefix + : 然后输入 setw synchronize-panes off
```

适用场景：同时操作多台服务器。

### 6.2 发送命令到指定窗格

```bash
# 向 myproject 会话的第 1 个窗口发送命令
tmux send-keys -t myproject:1 "tail -f /var/log/app.log" Enter

# 向当前窗口的第 0 个窗格发送命令
tmux send-keys -t 0 "echo hello" Enter
```

### 6.3 保存窗格历史到文件

```bash
# 保存当前窗格的输出历史
tmux capture-pane -pS - > ~/tmux_output.txt

# 保存指定行数的历史（最近 5000 行）
tmux capture-pane -pS -5000 > ~/tmux_output.txt
```

### 6.4 显示时钟

```bash
prefix + t
# 按任意键退出
```

### 6.5 显示窗格编号

```bash
prefix + q
# 显示编号后，快速按数字可切换到对应窗格
```

---

## 7. 配置文件 (~/.tmux.conf)

tmux 的配置文件为 `~/.tmux.conf`，修改后需重新加载。

### 7.1 重新加载配置

```bash
# 命令行方式
tmux source-file ~/.tmux.conf

# 快捷键方式（需先在配置中绑定）
prefix + r    # 见下方配置示例
```

### 7.2 常用配置示例

```bash
# ========== 基础设置 ==========

# 修改前缀键为 Ctrl+a（更顺手）
set -g prefix C-a
unbind C-b
bind C-a send-prefix

# 设置 r 键重新加载配置
bind r source-file ~/.tmux.conf \; display "Config reloaded!"

# 开启鼠标支持（可以用鼠标选窗格、调大小、滚动）
set -g mouse on

# 从 1 开始编号窗口和窗格（默认从 0 开始）
set -g base-index 1
setw -g pane-base-index 1

# 窗口关闭后自动重新编号
set -g renumber-windows on

# 设置历史记录行数
set -g history-limit 50000

# 减少 ESC 延迟（对 vim 用户重要）
set -sg escape-time 0

# ========== 分屏快捷键优化 ==========

# 用 | 和 - 替代 " 和 %（更直观）
bind | split-window -h -c "#{pane_current_path}"
bind - split-window -v -c "#{pane_current_path}"
unbind '"'
unbind %

# 新窗口保持当前路径
bind c new-window -c "#{pane_current_path}"

# ========== 窗格切换优化 ==========

# 使用 Alt+方向键切换窗格（无需前缀键）
bind -n M-Left select-pane -L
bind -n M-Right select-pane -R
bind -n M-Up select-pane -U
bind -n M-Down select-pane -D

# 使用 vim 风格的 hjkl 切换窗格
bind h select-pane -L
bind j select-pane -D
bind k select-pane -U
bind l select-pane -R

# ========== 复制模式 ==========

# 使用 vi 风格按键
setw -g mode-keys vi

# v 开始选择，y 复制
bind -T copy-mode-vi v send-keys -X begin-selection
bind -T copy-mode-vi y send-keys -X copy-selection-and-cancel

# ========== 外观 ==========

# 启用 256 色
set -g default-terminal "screen-256color"

# 状态栏样式
set -g status-style bg=black,fg=white
set -g status-left "[#S] "
set -g status-right "%Y-%m-%d %H:%M"

# 当前窗口高亮
setw -g window-status-current-style fg=black,bg=cyan
```

---

## 8. 快捷键速查表

### 会话

| 快捷键 | 功能 |
|--------|------|
| `prefix + d` | 分离会话 |
| `prefix + s` | 列出并选择会话 |
| `prefix + $` | 重命名会话 |
| `prefix + (` | 上一个会话 |
| `prefix + )` | 下一个会话 |

### 窗口

| 快捷键 | 功能 |
|--------|------|
| `prefix + c` | 新建窗口 |
| `prefix + n` | 下一个窗口 |
| `prefix + p` | 上一个窗口 |
| `prefix + 0-9` | 切换到指定窗口 |
| `prefix + w` | 窗口列表 |
| `prefix + ,` | 重命名窗口 |
| `prefix + &` | 关闭窗口 |

### 窗格

| 快捷键 | 功能 |
|--------|------|
| `prefix + "` | 水平分屏 |
| `prefix + %` | 垂直分屏 |
| `prefix + 方向键` | 切换窗格 |
| `prefix + z` | 窗格缩放 |
| `prefix + x` | 关闭窗格 |
| `prefix + !` | 窗格转窗口 |
| `prefix + Space` | 切换布局 |
| `prefix + q` | 显示窗格编号 |
| `prefix + {` | 与上一个窗格交换 |
| `prefix + }` | 与下一个窗格交换 |
| `prefix + o` | 切到下一个窗格 |

### 复制模式

| 快捷键 | 功能 |
|--------|------|
| `prefix + [` | 进入复制模式 |
| `prefix + ]` | 粘贴 |
| `prefix + =` | 选择缓冲区粘贴 |

### 其他

| 快捷键 | 功能 |
|--------|------|
| `prefix + :` | 进入命令模式 |
| `prefix + t` | 显示时钟 |
| `prefix + ?` | 列出所有快捷键 |
