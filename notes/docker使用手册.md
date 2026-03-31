# Docker 使用手册

Docker 是一个开源的容器化平台，允许你将应用及其依赖打包到一个轻量级、可移植的容器中运行。

---

## 核心概念

| 概念 | 说明 |
|------|------|
| **Image（镜像）** | 只读模板，包含运行应用所需的一切（代码、运行时、库、配置） |
| **Container（容器）** | 镜像的运行实例，相互隔离，可启停销毁 |
| **Dockerfile** | 构建镜像的脚本文件，定义镜像的每一层 |
| **Registry（仓库）** | 存储和分发镜像的服务，如 Docker Hub |
| **Volume（卷）** | 持久化数据存储，独立于容器生命周期 |
| **Network（网络）** | 容器间通信的虚拟网络 |
| **Compose** | 通过 YAML 文件定义和运行多容器应用 |

---

## 1. Docker 安装与基础信息

### 1.1 查看 Docker 版本

```bash
# 简要版本
docker --version
# 输出示例：Docker version 24.0.7, build afdd53b

# 详细版本信息（客户端+服务端）
docker version
```

### 1.2 查看系统信息

```bash
docker info
# 显示容器数量、镜像数量、存储驱动、内核版本等
```

### 1.3 查看磁盘使用

```bash
docker system df
# 输出示例：
# TYPE            TOTAL   ACTIVE  SIZE      RECLAIMABLE
# Images          15      5       4.2GB     2.8GB (66%)
# Containers      8       3       120MB     80MB (66%)
# Local Volumes   6       4       500MB     100MB (20%)
# Build Cache     20      0       800MB     800MB

# 详细信息
docker system df -v
```

---

## 2. 镜像管理（Image）

### 2.1 搜索镜像

```bash
# 在 Docker Hub 搜索镜像
docker search nginx

# 限制结果数量
docker search nginx --limit 5

# 只显示官方镜像
docker search nginx --filter is-official=true
```

### 2.2 拉取镜像

```bash
# 拉取最新版本
docker pull nginx

# 拉取指定版本
docker pull nginx:1.25

# 拉取指定平台的镜像
docker pull --platform linux/amd64 python:3.12

# 从其他仓库拉取
docker pull ghcr.io/owner/repo:tag
```

### 2.3 查看本地镜像

```bash
# 列出所有镜像
docker images
# 等价写法
docker image ls

# 只显示镜像 ID
docker images -q

# 按名称过滤
docker images nginx

# 按条件过滤（悬空镜像）
docker images -f dangling=true

# 自定义输出格式
docker images --format "{{.Repository}}:{{.Tag}} — {{.Size}}"
```

### 2.4 查看镜像详情

```bash
# 查看镜像完整信息（JSON）
docker inspect nginx:latest

# 查看镜像历史（每一层的构建命令）
docker history nginx:latest

# 查看镜像的指定字段
docker inspect nginx:latest --format '{{.Config.ExposedPorts}}'
```

### 2.5 标记镜像

```bash
# 给镜像打标签
docker tag nginx:latest myregistry.com/nginx:v1.0

# 为推送做准备
docker tag myapp:latest username/myapp:1.0
```

### 2.6 推送镜像

```bash
# 登录 Docker Hub
docker login

# 推送镜像
docker push username/myapp:1.0

# 登录其他仓库
docker login ghcr.io
docker push ghcr.io/owner/myapp:1.0

# 登出
docker logout
```

### 2.7 删除镜像

```bash
# 删除指定镜像
docker rmi nginx:latest

# 强制删除（即使有容器引用）
docker rmi -f nginx:latest

# 删除所有悬空镜像（无标签的）
docker image prune

# 删除所有未被容器使用的镜像
docker image prune -a

# 删除所有镜像
docker rmi $(docker images -q)
```

### 2.8 导出和导入镜像

```bash
# 导出镜像为 tar 文件
docker save nginx:latest -o nginx.tar
docker save nginx:latest > nginx.tar

# 导入镜像
docker load -i nginx.tar
docker load < nginx.tar
```

---

## 3. 容器管理（Container）

### 3.1 创建并运行容器

```bash
# 最基本的运行
docker run nginx

# 后台运行（-d = detach）
docker run -d nginx

# 指定容器名称
docker run -d --name my-nginx nginx

# 端口映射（宿主机:容器）
docker run -d -p 8080:80 nginx
# 访问 http://localhost:8080 即可

# 多端口映射
docker run -d -p 8080:80 -p 8443:443 nginx

# 随机端口映射
docker run -d -P nginx

# 交互式运行（-i 交互 + -t 终端）
docker run -it ubuntu:22.04 /bin/bash

# 运行后自动删除容器
docker run --rm -it python:3.12 python -c "print('hello')"

# 设置环境变量
docker run -d -e MYSQL_ROOT_PASSWORD=secret -e MYSQL_DATABASE=mydb mysql:8

# 从文件读取环境变量
docker run -d --env-file .env nginx

# 挂载目录（宿主机路径:容器路径）
docker run -d -v /home/user/html:/usr/share/nginx/html nginx
docker run -d -v $(pwd)/data:/app/data myapp

# 只读挂载
docker run -d -v /config:/app/config:ro nginx

# 限制资源
docker run -d --memory=512m --cpus=1.5 nginx

# 设置重启策略
docker run -d --restart=always nginx        # 总是重启
docker run -d --restart=unless-stopped nginx # 除非手动停止
docker run -d --restart=on-failure:3 nginx   # 失败时最多重试 3 次

# 设置工作目录
docker run -d -w /app node:20 npm start

# 指定网络
docker run -d --network my-network nginx

# 综合示例：运行一个完整的 Web 应用
docker run -d \
  --name my-webapp \
  -p 3000:3000 \
  -v $(pwd)/src:/app/src \
  -e NODE_ENV=production \
  --memory=1g \
  --restart=unless-stopped \
  node:20 npm start
```

### 3.2 查看容器

```bash
# 查看运行中的容器
docker ps

# 查看所有容器（含已停止的）
docker ps -a

# 只显示容器 ID
docker ps -q

# 查看最近创建的 5 个容器
docker ps -n 5

# 按条件过滤
docker ps -f status=exited
docker ps -f name=nginx

# 自定义输出格式
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

# 查看容器详细信息
docker inspect my-nginx

# 查看容器端口映射
docker port my-nginx
```

### 3.3 启停容器

```bash
# 停止容器（发送 SIGTERM，默认 10 秒后 SIGKILL）
docker stop my-nginx

# 指定等待时间后强制停止
docker stop -t 30 my-nginx

# 启动已停止的容器
docker start my-nginx

# 重启容器
docker restart my-nginx

# 暂停容器（冻结进程）
docker pause my-nginx

# 恢复暂停的容器
docker unpause my-nginx

# 强制杀死容器（立即 SIGKILL）
docker kill my-nginx

# 停止所有运行中的容器
docker stop $(docker ps -q)
```

### 3.4 进入容器

```bash
# 在运行中的容器内执行命令
docker exec my-nginx ls /etc/nginx

# 进入容器的交互式 shell
docker exec -it my-nginx /bin/bash
docker exec -it my-nginx /bin/sh    # Alpine 等轻量镜像用 sh

# 以 root 用户进入
docker exec -it -u root my-nginx /bin/bash

# 设置环境变量后执行
docker exec -e MY_VAR=hello my-nginx env

# 在指定工作目录下执行
docker exec -w /etc/nginx my-nginx cat nginx.conf
```

### 3.5 查看容器日志

```bash
# 查看全部日志
docker logs my-nginx

# 实时跟踪日志（类似 tail -f）
docker logs -f my-nginx

# 显示时间戳
docker logs -t my-nginx

# 只看最近 100 行
docker logs --tail 100 my-nginx

# 查看指定时间之后的日志
docker logs --since 2026-03-31T10:00:00 my-nginx
docker logs --since 30m my-nginx    # 最近 30 分钟

# 组合使用
docker logs -f --tail 50 -t my-nginx
```

### 3.6 查看容器资源占用

```bash
# 实时查看所有容器的 CPU/内存/网络/IO
docker stats

# 查看指定容器
docker stats my-nginx

# 只获取一次快照（不实时刷新）
docker stats --no-stream

# 自定义输出
docker stats --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}"
```

### 3.7 查看容器内进程

```bash
docker top my-nginx
```

### 3.8 容器与宿主机之间复制文件

```bash
# 从宿主机复制到容器
docker cp ./index.html my-nginx:/usr/share/nginx/html/

# 从容器复制到宿主机
docker cp my-nginx:/etc/nginx/nginx.conf ./nginx.conf

# 复制整个目录
docker cp my-nginx:/var/log/nginx/ ./nginx-logs/
```

### 3.9 查看容器文件系统变更

```bash
# 查看容器相对于镜像的文件修改
docker diff my-nginx
# A = 新增, C = 修改, D = 删除
```

### 3.10 删除容器

```bash
# 删除已停止的容器
docker rm my-nginx

# 强制删除运行中的容器
docker rm -f my-nginx

# 删除容器并移除其卷
docker rm -v my-nginx

# 删除所有已停止的容器
docker container prune

# 删除所有容器
docker rm -f $(docker ps -aq)
```

### 3.11 将容器保存为镜像

```bash
# 将容器的当前状态提交为新镜像
docker commit my-nginx my-nginx-custom:v1

# 附带提交信息
docker commit -m "added custom config" my-nginx my-nginx-custom:v1

# 导出容器为 tar（不保留历史层）
docker export my-nginx > container.tar

# 从 tar 导入为镜像
docker import container.tar myimage:latest
```

---

## 4. Dockerfile 构建镜像

### 4.1 Dockerfile 常用指令

```dockerfile
# ===== 基础 Python Web 应用示例 =====

# 基础镜像
FROM python:3.12-slim

# 镜像元数据
LABEL maintainer="dev@example.com"
LABEL version="1.0"

# 设置环境变量
ENV PYTHONUNBUFFERED=1
ENV APP_HOME=/app

# 设置工作目录
WORKDIR $APP_HOME

# 复制依赖文件（利用缓存，先复制变化少的文件）
COPY requirements.txt .

# 执行命令（安装依赖）
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY . .

# 创建非 root 用户
RUN useradd -m appuser
USER appuser

# 声明容器监听的端口（文档作用）
EXPOSE 8000

# 健康检查
HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# 容器启动命令
CMD ["python", "app.py"]
```

```dockerfile
# ===== 多阶段构建示例（Go 应用）=====

# 阶段一：构建
FROM golang:1.22 AS builder
WORKDIR /build
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 go build -o /app/server .

# 阶段二：运行（最终镜像极小）
FROM alpine:3.19
RUN apk --no-cache add ca-certificates
WORKDIR /app
COPY --from=builder /app/server .
EXPOSE 8080
CMD ["./server"]
```

```dockerfile
# ===== Node.js 应用示例 =====

FROM node:20-alpine
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci --only=production
COPY . .
EXPOSE 3000
CMD ["node", "server.js"]
```

### 4.2 构建镜像

```bash
# 在当前目录构建（. 表示构建上下文）
docker build -t myapp:1.0 .

# 指定 Dockerfile 路径
docker build -t myapp:1.0 -f docker/Dockerfile .

# 传递构建参数
docker build --build-arg NODE_ENV=production -t myapp:1.0 .

# 不使用缓存
docker build --no-cache -t myapp:1.0 .

# 多平台构建
docker buildx build --platform linux/amd64,linux/arm64 -t myapp:1.0 .

# 指定构建目标阶段（多阶段构建时）
docker build --target builder -t myapp:builder .

# 查看构建过程中每一层的大小
docker history myapp:1.0
```

---

## 5. 数据卷（Volume）

### 5.1 创建和管理卷

```bash
# 创建命名卷
docker volume create mydata

# 查看所有卷
docker volume ls

# 查看卷详情
docker volume inspect mydata

# 删除卷
docker volume rm mydata

# 删除所有未使用的卷
docker volume prune
```

### 5.2 使用卷

```bash
# 使用命名卷（推荐）
docker run -d -v mydata:/var/lib/mysql mysql:8

# 使用 --mount 语法（更明确）
docker run -d --mount source=mydata,target=/var/lib/mysql mysql:8

# 只读卷
docker run -d --mount source=mydata,target=/data,readonly nginx

# 使用 tmpfs 挂载（数据存于内存，容器停止即消失）
docker run -d --tmpfs /tmp:rw,size=100m nginx

# 共享卷（多个容器共享同一个卷）
docker run -d --name writer -v shared-data:/data alpine sh -c "while true; do date >> /data/log.txt; sleep 5; done"
docker run -d --name reader -v shared-data:/data:ro alpine tail -f /data/log.txt
```

### 5.3 备份和恢复卷

```bash
# 备份：启动临时容器，把卷内容打包到宿主机
docker run --rm \
  -v mydata:/source:ro \
  -v $(pwd):/backup \
  alpine tar czf /backup/mydata-backup.tar.gz -C /source .

# 恢复：启动临时容器，把备份解压到卷中
docker run --rm \
  -v mydata:/target \
  -v $(pwd):/backup \
  alpine tar xzf /backup/mydata-backup.tar.gz -C /target
```

---

## 6. 网络管理（Network）

### 6.1 查看和创建网络

```bash
# 查看所有网络
docker network ls

# 创建自定义桥接网络
docker network create mynet

# 创建时指定子网和网关
docker network create --subnet=172.20.0.0/16 --gateway=172.20.0.1 mynet

# 创建 overlay 网络（Swarm 模式下跨主机通信）
docker network create --driver overlay my-overlay

# 查看网络详情
docker network inspect mynet

# 删除网络
docker network rm mynet

# 删除所有未使用的网络
docker network prune
```

### 6.2 容器连接网络

```bash
# 运行容器时指定网络
docker run -d --name web --network mynet nginx

# 将运行中的容器连接到网络
docker network connect mynet my-nginx

# 连接时指定 IP
docker network connect --ip 172.20.0.10 mynet my-nginx

# 断开容器与网络的连接
docker network disconnect mynet my-nginx
```

### 6.3 容器间通信

```bash
# 同一自定义网络中的容器可以通过容器名互相访问
docker network create app-net

docker run -d --name db --network app-net \
  -e POSTGRES_PASSWORD=secret postgres:16

docker run -d --name api --network app-net \
  -e DATABASE_URL=postgresql://postgres:secret@db:5432/postgres \
  myapi:latest
# api 容器可以通过 "db" 这个主机名访问数据库容器
```

---

## 7. Docker Compose

Docker Compose 用于定义和运行多容器应用，配置文件为 `docker-compose.yml`（或 `compose.yml`）。

### 7.1 compose.yml 示例

```yaml
# ===== 典型 Web 应用（前端 + 后端 + 数据库 + 缓存）=====

services:
  # Nginx 反向代理
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
    depends_on:
      - api
    restart: unless-stopped

  # 后端 API
  api:
    build:
      context: ./api
      dockerfile: Dockerfile
    environment:
      - DATABASE_URL=postgresql://postgres:secret@db:5432/mydb
      - REDIS_URL=redis://cache:6379
    depends_on:
      db:
        condition: service_healthy
      cache:
        condition: service_started
    restart: unless-stopped

  # 数据库
  db:
    image: postgres:16
    environment:
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: secret
      POSTGRES_DB: mydb
    volumes:
      - pgdata:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 10s
      timeout: 5s
      retries: 5
    restart: unless-stopped

  # 缓存
  cache:
    image: redis:7-alpine
    restart: unless-stopped

volumes:
  pgdata:
```

### 7.2 Compose 常用命令

```bash
# 启动所有服务（后台运行）
docker compose up -d

# 启动并强制重新构建镜像
docker compose up -d --build

# 只启动指定服务
docker compose up -d api db

# 查看服务状态
docker compose ps

# 查看日志
docker compose logs
docker compose logs -f api         # 实时跟踪 api 服务
docker compose logs --tail 50 db   # db 服务最近 50 行

# 在服务容器中执行命令
docker compose exec api /bin/bash
docker compose exec db psql -U postgres

# 运行一次性命令（启动新容器）
docker compose run --rm api python manage.py migrate

# 停止所有服务
docker compose stop

# 停止并移除容器、网络
docker compose down

# 停止并移除容器、网络、卷（慎用，会丢数据）
docker compose down -v

# 停止并移除容器、网络、镜像
docker compose down --rmi all

# 重启指定服务
docker compose restart api

# 扩展服务实例数
docker compose up -d --scale api=3

# 查看服务资源占用
docker compose top

# 拉取所有服务的最新镜像
docker compose pull

# 验证 compose 文件语法
docker compose config
```

---

## 8. 系统清理

```bash
# 删除所有已停止的容器
docker container prune

# 删除所有悬空镜像
docker image prune

# 删除所有未使用的镜像
docker image prune -a

# 删除所有未使用的卷
docker volume prune

# 删除所有未使用的网络
docker network prune

# 一键清理（容器 + 镜像 + 网络 + 构建缓存）
docker system prune

# 一键清理（包含未使用的卷）
docker system prune -a --volumes

# 清理构建缓存
docker builder prune
docker builder prune --all    # 清理全部构建缓存
```

---

## 9. 常用实战场景

### 9.1 快速启动常用服务

```bash
# MySQL
docker run -d --name mysql \
  -p 3306:3306 \
  -e MYSQL_ROOT_PASSWORD=secret \
  -e MYSQL_DATABASE=mydb \
  -v mysql-data:/var/lib/mysql \
  mysql:8

# PostgreSQL
docker run -d --name postgres \
  -p 5432:5432 \
  -e POSTGRES_PASSWORD=secret \
  -e POSTGRES_DB=mydb \
  -v pg-data:/var/lib/postgresql/data \
  postgres:16

# Redis
docker run -d --name redis \
  -p 6379:6379 \
  redis:7-alpine

# MongoDB
docker run -d --name mongo \
  -p 27017:27017 \
  -e MONGO_INITDB_ROOT_USERNAME=admin \
  -e MONGO_INITDB_ROOT_PASSWORD=secret \
  -v mongo-data:/data/db \
  mongo:7

# Nginx（静态文件服务器）
docker run -d --name nginx \
  -p 80:80 \
  -v $(pwd)/html:/usr/share/nginx/html:ro \
  nginx:alpine

# Elasticsearch
docker run -d --name es \
  -p 9200:9200 \
  -e discovery.type=single-node \
  -e ES_JAVA_OPTS="-Xms512m -Xmx512m" \
  -v es-data:/usr/share/elasticsearch/data \
  elasticsearch:8.12.0

# RabbitMQ（带管理界面）
docker run -d --name rabbitmq \
  -p 5672:5672 \
  -p 15672:15672 \
  rabbitmq:3-management
```

### 9.2 调试容器问题

```bash
# 查看容器退出原因
docker inspect my-container --format '{{.State.ExitCode}} {{.State.Error}}'

# 查看容器最后的日志
docker logs --tail 50 my-container

# 查看容器的事件
docker events --filter container=my-container

# 用相同镜像启动一个调试容器
docker run --rm -it --entrypoint /bin/sh myapp:latest

# 查看容器内的文件系统变更
docker diff my-container

# 查看容器的资源限制
docker inspect my-container --format '{{.HostConfig.Memory}} {{.HostConfig.NanoCpus}}'
```

### 9.3 镜像瘦身技巧

```bash
# 查看镜像各层大小，找出占用最大的层
docker history myapp:latest --format "{{.Size}}\t{{.CreatedBy}}" --no-trunc

# 使用 Alpine 基础镜像（通常只有 5MB）
# FROM python:3.12       → ~1GB
# FROM python:3.12-slim  → ~150MB
# FROM python:3.12-alpine → ~50MB

# 使用 .dockerignore 排除不必要的文件
# .dockerignore 示例：
# .git
# node_modules
# __pycache__
# *.pyc
# .env
# README.md
# tests/
```

---

## 10. 快捷操作速查表

### 容器生命周期

| 命令 | 功能 |
|------|------|
| `docker run` | 创建并启动容器 |
| `docker start` | 启动已停止的容器 |
| `docker stop` | 停止容器 |
| `docker restart` | 重启容器 |
| `docker pause/unpause` | 暂停/恢复容器 |
| `docker kill` | 强制终止容器 |
| `docker rm` | 删除容器 |

### 容器信息

| 命令 | 功能 |
|------|------|
| `docker ps` | 查看运行中的容器 |
| `docker ps -a` | 查看所有容器 |
| `docker logs` | 查看日志 |
| `docker stats` | 资源占用 |
| `docker top` | 容器内进程 |
| `docker inspect` | 详细信息 |
| `docker port` | 端口映射 |
| `docker diff` | 文件变更 |

### 容器交互

| 命令 | 功能 |
|------|------|
| `docker exec -it <name> bash` | 进入容器 |
| `docker cp` | 复制文件 |
| `docker attach` | 附加到容器主进程 |

### 镜像操作

| 命令 | 功能 |
|------|------|
| `docker pull` | 拉取镜像 |
| `docker push` | 推送镜像 |
| `docker build` | 构建镜像 |
| `docker images` | 查看镜像列表 |
| `docker rmi` | 删除镜像 |
| `docker tag` | 标记镜像 |
| `docker save/load` | 导出/导入镜像（保留层） |
| `docker export/import` | 导出/导入容器（扁平化） |

### Compose 操作

| 命令 | 功能 |
|------|------|
| `docker compose up -d` | 启动所有服务 |
| `docker compose down` | 停止并移除 |
| `docker compose ps` | 查看服务状态 |
| `docker compose logs -f` | 实时日志 |
| `docker compose exec` | 进入服务容器 |
| `docker compose build` | 构建服务镜像 |
| `docker compose pull` | 拉取服务镜像 |

### 常用 run 参数

| 参数 | 功能 | 示例 |
|------|------|------|
| `-d` | 后台运行 | `docker run -d nginx` |
| `-p` | 端口映射 | `-p 8080:80` |
| `-v` | 挂载卷 | `-v ./data:/data` |
| `-e` | 环境变量 | `-e KEY=value` |
| `--name` | 容器名称 | `--name myapp` |
| `--rm` | 退出后自动删除 | `docker run --rm alpine echo hi` |
| `-it` | 交互式终端 | `docker run -it ubuntu bash` |
| `--network` | 指定网络 | `--network mynet` |
| `--restart` | 重启策略 | `--restart=unless-stopped` |
| `--memory` | 内存限制 | `--memory=512m` |
| `--cpus` | CPU 限制 | `--cpus=2` |
| `-w` | 工作目录 | `-w /app` |
| `-u` | 运行用户 | `-u 1000:1000` |
