# Kubernetes (k8s) 使用手册

Kubernetes 是一个开源的容器编排平台，用于自动化容器的部署、扩缩容和管理。

---

## 核心概念

| 概念 | 说明 |
|------|------|
| **Cluster（集群）** | 由 Master 节点和 Worker 节点组成的整体 |
| **Node（节点）** | 集群中的一台机器（物理机或虚拟机） |
| **Pod** | 最小调度单元，包含一个或多个容器 |
| **Deployment** | 管理 Pod 的副本数量、滚动更新和回滚 |
| **Service** | 为一组 Pod 提供稳定的网络访问入口 |
| **Namespace** | 资源的逻辑隔离空间 |
| **ConfigMap / Secret** | 配置和敏感信息管理 |
| **Ingress** | HTTP/HTTPS 路由规则，对外暴露服务 |
| **PV / PVC** | 持久化存储卷和存储声明 |
| **StatefulSet** | 有状态应用的管理（如数据库） |
| **DaemonSet** | 确保每个节点运行一个 Pod 副本 |
| **Job / CronJob** | 一次性任务 / 定时任务 |
| **HPA** | 水平自动扩缩容 |

---

## 1. 集群与上下文管理

### 1.1 查看集群信息

```bash
# 查看集群基本信息
kubectl cluster-info
# 输出示例：
# Kubernetes control plane is running at https://192.168.1.100:6443
# CoreDNS is running at https://192.168.1.100:6443/api/v1/namespaces/kube-system/services/kube-dns:dns/proxy

# 查看集群版本
kubectl version
# 只看客户端版本
kubectl version --client

# 查看 API 资源列表
kubectl api-resources

# 查看 API 版本
kubectl api-versions
```

### 1.2 上下文管理（多集群切换）

```bash
# 查看所有上下文
kubectl config get-contexts
# 输出示例：
# CURRENT   NAME        CLUSTER     AUTHINFO    NAMESPACE
# *         dev         dev-cluster dev-admin   default
#           staging     stg-cluster stg-admin   default
#           production  prd-cluster prd-admin   default

# 查看当前上下文
kubectl config current-context

# 切换上下文
kubectl config use-context production

# 设置默认命名空间
kubectl config set-context --current --namespace=my-namespace

# 查看完整 kubeconfig
kubectl config view

# 合并多个 kubeconfig
KUBECONFIG=~/.kube/config:~/.kube/config-staging kubectl config view --merge --flatten > ~/.kube/merged-config
```

---

## 2. 命名空间（Namespace）

```bash
# 查看所有命名空间
kubectl get namespaces
kubectl get ns

# 创建命名空间
kubectl create namespace dev
kubectl create ns staging

# 删除命名空间（会删除其下所有资源，慎用）
kubectl delete namespace dev

# 查看指定命名空间下的资源
kubectl get pods -n kube-system

# 查看所有命名空间下的资源
kubectl get pods --all-namespaces
kubectl get pods -A
```

---

## 3. Pod 管理

### 3.1 查看 Pod

```bash
# 查看当前命名空间的 Pod
kubectl get pods
kubectl get po

# 查看所有命名空间的 Pod
kubectl get pods -A

# 显示更多信息（IP、节点等）
kubectl get pods -o wide

# 按标签过滤
kubectl get pods -l app=nginx
kubectl get pods -l "app=nginx,env=production"
kubectl get pods -l "app in (nginx,apache)"

# 按字段过滤
kubectl get pods --field-selector status.phase=Running
kubectl get pods --field-selector spec.nodeName=worker-01

# 监听实时变化
kubectl get pods -w

# 自定义输出列
kubectl get pods -o custom-columns=NAME:.metadata.name,STATUS:.status.phase,NODE:.spec.nodeName

# JSON/YAML 格式输出
kubectl get pod my-pod -o yaml
kubectl get pod my-pod -o json

# 使用 JSONPath
kubectl get pods -o jsonpath='{.items[*].metadata.name}'
kubectl get pods -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.status.phase}{"\n"}{end}'

# 排序
kubectl get pods --sort-by=.metadata.creationTimestamp
kubectl get pods --sort-by=.status.containerStatuses[0].restartCount
```

### 3.2 查看 Pod 详情

```bash
# 查看 Pod 的详细描述（事件、条件、挂载等）
kubectl describe pod my-pod

# 查看 Pod 的 YAML 定义
kubectl get pod my-pod -o yaml

# 查看 Pod 中容器的资源使用（需 metrics-server）
kubectl top pod my-pod
kubectl top pods                    # 所有 Pod
kubectl top pods --containers       # 按容器显示
kubectl top pods --sort-by=memory   # 按内存排序
```

### 3.3 创建 Pod

```bash
# 快速运行一个 Pod
kubectl run my-nginx --image=nginx:alpine

# 运行并暴露端口
kubectl run my-nginx --image=nginx --port=80

# 运行并设置环境变量
kubectl run my-app --image=myapp:latest --env="DB_HOST=mysql" --env="DB_PORT=3306"

# 运行一次性交互式 Pod（退出后自动删除）
kubectl run debug --rm -it --image=busybox -- /bin/sh
kubectl run debug --rm -it --image=ubuntu:22.04 -- bash
kubectl run debug --rm -it --image=nicolaka/netshoot -- bash    # 网络调试工具箱

# 运行一次性命令
kubectl run test --rm -it --image=curlimages/curl --restart=Never -- curl http://my-service:8080

# 试运行，只生成 YAML 不实际创建
kubectl run my-nginx --image=nginx --dry-run=client -o yaml > pod.yaml
```

### 3.4 Pod YAML 示例

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: my-app
  labels:
    app: my-app
    env: dev
spec:
  containers:
    - name: app
      image: python:3.12-slim
      ports:
        - containerPort: 8000
      env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: db-secret
              key: url
      resources:
        requests:
          memory: "128Mi"
          cpu: "250m"
        limits:
          memory: "512Mi"
          cpu: "1"
      livenessProbe:
        httpGet:
          path: /health
          port: 8000
        initialDelaySeconds: 10
        periodSeconds: 30
      readinessProbe:
        httpGet:
          path: /ready
          port: 8000
        initialDelaySeconds: 5
        periodSeconds: 10
      volumeMounts:
        - name: config-vol
          mountPath: /app/config
  volumes:
    - name: config-vol
      configMap:
        name: app-config
  restartPolicy: Always
```

### 3.5 进入 Pod / 执行命令

```bash
# 在 Pod 内执行命令
kubectl exec my-pod -- ls /app
kubectl exec my-pod -- cat /etc/hosts

# 进入交互式 shell
kubectl exec -it my-pod -- /bin/bash
kubectl exec -it my-pod -- /bin/sh

# 多容器 Pod 指定容器
kubectl exec -it my-pod -c sidecar -- /bin/sh

# 在 Pod 中执行带管道的命令
kubectl exec my-pod -- sh -c "ps aux | grep python"
```

### 3.6 查看 Pod 日志

```bash
# 查看日志
kubectl logs my-pod

# 实时跟踪日志
kubectl logs -f my-pod

# 查看最近 100 行
kubectl logs --tail=100 my-pod

# 查看最近 1 小时的日志
kubectl logs --since=1h my-pod

# 查看指定时间之后的日志
kubectl logs --since-time="2026-03-31T10:00:00Z" my-pod

# 多容器 Pod 指定容器
kubectl logs my-pod -c sidecar

# 查看所有容器的日志
kubectl logs my-pod --all-containers

# 查看前一个（崩溃的）容器的日志
kubectl logs my-pod --previous

# 按标签查看多个 Pod 的日志
kubectl logs -l app=nginx --all-containers

# 带前缀显示（区分多个 Pod）
kubectl logs -l app=nginx --prefix
```

### 3.7 文件复制

```bash
# 从本地复制到 Pod
kubectl cp ./config.yaml my-pod:/app/config.yaml

# 从 Pod 复制到本地
kubectl cp my-pod:/app/logs/ ./pod-logs/

# 指定命名空间
kubectl cp dev/my-pod:/data/dump.sql ./dump.sql

# 指定容器
kubectl cp my-pod:/app/data ./data -c app
```

### 3.8 端口转发

```bash
# 将 Pod 的端口转发到本地
kubectl port-forward my-pod 8080:80
# 访问 http://localhost:8080 即转发到 Pod 的 80 端口

# 转发多个端口
kubectl port-forward my-pod 8080:80 8443:443

# 转发 Service 的端口
kubectl port-forward svc/my-service 8080:80

# 绑定到所有网络接口（允许外部访问）
kubectl port-forward --address 0.0.0.0 my-pod 8080:80
```

### 3.9 删除 Pod

```bash
# 删除指定 Pod
kubectl delete pod my-pod

# 强制删除（不等待优雅终止）
kubectl delete pod my-pod --force --grace-period=0

# 按标签删除
kubectl delete pods -l app=test

# 删除命名空间下所有 Pod
kubectl delete pods --all -n dev
```

---

## 4. Deployment 管理

### 4.1 创建 Deployment

```bash
# 快速创建
kubectl create deployment my-nginx --image=nginx:alpine --replicas=3

# 生成 YAML（不实际创建）
kubectl create deployment my-app --image=myapp:1.0 --replicas=3 --dry-run=client -o yaml > deployment.yaml

# 从 YAML 文件创建
kubectl apply -f deployment.yaml
```

### 4.2 Deployment YAML 示例

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: my-app
  labels:
    app: my-app
spec:
  replicas: 3
  selector:
    matchLabels:
      app: my-app
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1          # 滚动更新时最多额外创建 1 个 Pod
      maxUnavailable: 0     # 滚动更新时不允许不可用 Pod
  template:
    metadata:
      labels:
        app: my-app
    spec:
      containers:
        - name: app
          image: myapp:1.0
          ports:
            - containerPort: 8080
          resources:
            requests:
              memory: "256Mi"
              cpu: "250m"
            limits:
              memory: "512Mi"
              cpu: "500m"
          livenessProbe:
            httpGet:
              path: /health
              port: 8080
            initialDelaySeconds: 15
            periodSeconds: 20
          readinessProbe:
            httpGet:
              path: /ready
              port: 8080
            initialDelaySeconds: 5
            periodSeconds: 10
```

### 4.3 查看 Deployment

```bash
# 查看 Deployment 列表
kubectl get deployments
kubectl get deploy

# 查看详细描述
kubectl describe deployment my-app

# 查看 Deployment 关联的 ReplicaSet
kubectl get rs -l app=my-app

# 查看滚动更新状态
kubectl rollout status deployment/my-app
```

### 4.4 更新 Deployment

```bash
# 更新镜像版本（触发滚动更新）
kubectl set image deployment/my-app app=myapp:2.0

# 编辑 Deployment（打开编辑器）
kubectl edit deployment my-app

# 更新环境变量
kubectl set env deployment/my-app DB_HOST=new-mysql

# 更新资源限制
kubectl set resources deployment/my-app -c app --limits=memory=1Gi,cpu=1 --requests=memory=256Mi,cpu=250m

# 通过 patch 更新
kubectl patch deployment my-app -p '{"spec":{"replicas":5}}'

# 暂停滚动更新（做多项修改后再统一更新）
kubectl rollout pause deployment/my-app
kubectl set image deployment/my-app app=myapp:2.0
kubectl set env deployment/my-app LOG_LEVEL=debug
kubectl rollout resume deployment/my-app
```

### 4.5 扩缩容

```bash
# 手动扩缩容
kubectl scale deployment my-app --replicas=5

# 缩容到 0（停止所有 Pod 但保留 Deployment）
kubectl scale deployment my-app --replicas=0

# 条件扩容（当前副本数为 3 时才执行）
kubectl scale deployment my-app --replicas=5 --current-replicas=3
```

### 4.6 回滚

```bash
# 查看更新历史
kubectl rollout history deployment/my-app

# 查看指定版本的详情
kubectl rollout history deployment/my-app --revision=2

# 回滚到上一个版本
kubectl rollout undo deployment/my-app

# 回滚到指定版本
kubectl rollout undo deployment/my-app --to-revision=2

# 重启所有 Pod（不改变配置）
kubectl rollout restart deployment/my-app
```

### 4.7 删除 Deployment

```bash
# 删除 Deployment（同时删除其管理的 Pod）
kubectl delete deployment my-app

# 从文件删除
kubectl delete -f deployment.yaml
```

---

## 5. Service 管理

### 5.1 创建 Service

```bash
# 为 Deployment 创建 ClusterIP Service
kubectl expose deployment my-app --port=80 --target-port=8080

# 创建 NodePort Service
kubectl expose deployment my-app --port=80 --target-port=8080 --type=NodePort

# 创建 LoadBalancer Service（云环境）
kubectl expose deployment my-app --port=80 --target-port=8080 --type=LoadBalancer

# 生成 YAML
kubectl expose deployment my-app --port=80 --target-port=8080 --dry-run=client -o yaml > service.yaml
```

### 5.2 Service YAML 示例

```yaml
# ===== ClusterIP（集群内部访问）=====
apiVersion: v1
kind: Service
metadata:
  name: my-app-service
spec:
  type: ClusterIP
  selector:
    app: my-app
  ports:
    - port: 80            # Service 端口
      targetPort: 8080    # Pod 端口
      protocol: TCP

---
# ===== NodePort（节点端口暴露）=====
apiVersion: v1
kind: Service
metadata:
  name: my-app-nodeport
spec:
  type: NodePort
  selector:
    app: my-app
  ports:
    - port: 80
      targetPort: 8080
      nodePort: 30080     # 节点端口（30000-32767）

---
# ===== LoadBalancer（云负载均衡器）=====
apiVersion: v1
kind: Service
metadata:
  name: my-app-lb
spec:
  type: LoadBalancer
  selector:
    app: my-app
  ports:
    - port: 80
      targetPort: 8080

---
# ===== Headless Service（无 ClusterIP，直接返回 Pod IP）=====
apiVersion: v1
kind: Service
metadata:
  name: my-db-headless
spec:
  clusterIP: None
  selector:
    app: my-db
  ports:
    - port: 5432
      targetPort: 5432
```

### 5.3 查看和管理 Service

```bash
# 查看 Service
kubectl get services
kubectl get svc

# 查看详情
kubectl describe svc my-app-service

# 查看 Service 关联的 Endpoints
kubectl get endpoints my-app-service

# 删除 Service
kubectl delete svc my-app-service

# 在集群内测试 Service 连通性
kubectl run test --rm -it --image=curlimages/curl --restart=Never -- curl http://my-app-service:80
```

---

## 6. ConfigMap 与 Secret

### 6.1 ConfigMap

```bash
# 从命令行创建
kubectl create configmap app-config \
  --from-literal=DATABASE_HOST=mysql \
  --from-literal=DATABASE_PORT=3306 \
  --from-literal=LOG_LEVEL=info

# 从文件创建
kubectl create configmap nginx-config --from-file=nginx.conf
kubectl create configmap app-config --from-file=config/    # 整个目录

# 从 .env 文件创建
kubectl create configmap app-env --from-env-file=.env

# 查看 ConfigMap
kubectl get configmaps
kubectl get cm
kubectl describe cm app-config

# 查看 ConfigMap 数据
kubectl get cm app-config -o yaml

# 编辑 ConfigMap
kubectl edit cm app-config

# 删除 ConfigMap
kubectl delete cm app-config
```

**在 Pod 中使用 ConfigMap：**

```yaml
# 方式一：作为环境变量
spec:
  containers:
    - name: app
      image: myapp:1.0
      envFrom:
        - configMapRef:
            name: app-config    # 整个 ConfigMap 导入
      env:
        - name: SPECIFIC_KEY
          valueFrom:
            configMapKeyRef:
              name: app-config
              key: LOG_LEVEL    # 只导入指定 key

# 方式二：作为文件挂载
spec:
  containers:
    - name: app
      image: myapp:1.0
      volumeMounts:
        - name: config-volume
          mountPath: /app/config
  volumes:
    - name: config-volume
      configMap:
        name: app-config
```

### 6.2 Secret

```bash
# 创建 Generic Secret
kubectl create secret generic db-secret \
  --from-literal=username=admin \
  --from-literal=password='S3cr3t!@#'

# 从文件创建
kubectl create secret generic tls-cert --from-file=cert.pem --from-file=key.pem

# 创建 Docker Registry Secret（拉取私有镜像用）
kubectl create secret docker-registry my-registry \
  --docker-server=registry.example.com \
  --docker-username=user \
  --docker-password=pass \
  --docker-email=user@example.com

# 创建 TLS Secret
kubectl create secret tls my-tls --cert=path/to/cert.pem --key=path/to/key.pem

# 查看 Secret（值为 base64 编码）
kubectl get secrets
kubectl describe secret db-secret
kubectl get secret db-secret -o yaml

# 解码 Secret 值
kubectl get secret db-secret -o jsonpath='{.data.password}' | base64 -d

# 删除 Secret
kubectl delete secret db-secret
```

**在 Pod 中使用 Secret：**

```yaml
spec:
  containers:
    - name: app
      image: myapp:1.0
      env:
        - name: DB_PASSWORD
          valueFrom:
            secretKeyRef:
              name: db-secret
              key: password
      volumeMounts:
        - name: secret-volume
          mountPath: /app/secrets
          readOnly: true
  volumes:
    - name: secret-volume
      secret:
        secretName: db-secret
  imagePullSecrets:
    - name: my-registry    # 拉取私有镜像
```

---

## 7. Ingress

### 7.1 Ingress YAML 示例

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: my-ingress
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
spec:
  ingressClassName: nginx
  tls:
    - hosts:
        - myapp.example.com
      secretName: myapp-tls
  rules:
    - host: myapp.example.com
      http:
        paths:
          - path: /api
            pathType: Prefix
            backend:
              service:
                name: api-service
                port:
                  number: 80
          - path: /
            pathType: Prefix
            backend:
              service:
                name: frontend-service
                port:
                  number: 80
```

### 7.2 管理 Ingress

```bash
# 创建 Ingress
kubectl apply -f ingress.yaml

# 查看 Ingress
kubectl get ingress
kubectl get ing

# 查看详情（包含后端服务和路由规则）
kubectl describe ingress my-ingress

# 删除 Ingress
kubectl delete ingress my-ingress
```

---

## 8. 持久化存储（PV / PVC）

### 8.1 PersistentVolume 和 PersistentVolumeClaim

```yaml
# ===== PersistentVolume（管理员创建）=====
apiVersion: v1
kind: PersistentVolume
metadata:
  name: my-pv
spec:
  capacity:
    storage: 10Gi
  accessModes:
    - ReadWriteOnce       # RWO: 单节点读写
  persistentVolumeReclaimPolicy: Retain    # 释放后保留数据
  storageClassName: standard
  hostPath:               # 仅用于测试，生产环境用 NFS/云存储
    path: /data/pv

---
# ===== PersistentVolumeClaim（开发者创建）=====
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: my-pvc
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 5Gi
  storageClassName: standard

---
# ===== 在 Pod 中使用 =====
apiVersion: v1
kind: Pod
metadata:
  name: my-pod
spec:
  containers:
    - name: app
      image: myapp:1.0
      volumeMounts:
        - name: data
          mountPath: /app/data
  volumes:
    - name: data
      persistentVolumeClaim:
        claimName: my-pvc
```

### 8.2 管理存储资源

```bash
# 查看 PV
kubectl get pv

# 查看 PVC
kubectl get pvc

# 查看 StorageClass
kubectl get storageclass
kubectl get sc

# 查看 PV/PVC 绑定详情
kubectl describe pv my-pv
kubectl describe pvc my-pvc

# 删除 PVC
kubectl delete pvc my-pvc

# 扩容 PVC（StorageClass 需支持）
kubectl patch pvc my-pvc -p '{"spec":{"resources":{"requests":{"storage":"20Gi"}}}}'
```

---

## 9. StatefulSet（有状态应用）

### 9.1 StatefulSet YAML 示例

```yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: postgres
spec:
  serviceName: postgres-headless    # 必须关联 Headless Service
  replicas: 3
  selector:
    matchLabels:
      app: postgres
  template:
    metadata:
      labels:
        app: postgres
    spec:
      containers:
        - name: postgres
          image: postgres:16
          ports:
            - containerPort: 5432
          env:
            - name: POSTGRES_PASSWORD
              valueFrom:
                secretKeyRef:
                  name: pg-secret
                  key: password
          volumeMounts:
            - name: pg-data
              mountPath: /var/lib/postgresql/data
  volumeClaimTemplates:             # 每个 Pod 自动创建独立的 PVC
    - metadata:
        name: pg-data
      spec:
        accessModes: ["ReadWriteOnce"]
        storageClassName: standard
        resources:
          requests:
            storage: 10Gi
```

Pod 命名规则：`postgres-0`、`postgres-1`、`postgres-2`（有序、稳定）。

### 9.2 管理 StatefulSet

```bash
# 查看 StatefulSet
kubectl get statefulsets
kubectl get sts

# 扩缩容
kubectl scale sts postgres --replicas=5

# 滚动更新
kubectl rollout status sts postgres
kubectl rollout undo sts postgres

# 删除 StatefulSet（保留 PVC）
kubectl delete sts postgres

# 删除 StatefulSet 及关联的 PVC
kubectl delete sts postgres
kubectl delete pvc -l app=postgres
```

---

## 10. DaemonSet（每节点一个 Pod）

```yaml
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: fluentd
spec:
  selector:
    matchLabels:
      app: fluentd
  template:
    metadata:
      labels:
        app: fluentd
    spec:
      containers:
        - name: fluentd
          image: fluentd:latest
          volumeMounts:
            - name: varlog
              mountPath: /var/log
      volumes:
        - name: varlog
          hostPath:
            path: /var/log
      tolerations:                  # 允许调度到 Master 节点
        - key: node-role.kubernetes.io/control-plane
          effect: NoSchedule
```

```bash
# 查看 DaemonSet
kubectl get daemonsets
kubectl get ds

# 查看详情
kubectl describe ds fluentd
```

---

## 11. Job 与 CronJob

### 11.1 Job（一次性任务）

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: db-migration
spec:
  backoffLimit: 3           # 失败重试次数
  activeDeadlineSeconds: 600 # 超时时间（秒）
  template:
    spec:
      containers:
        - name: migrate
          image: myapp:1.0
          command: ["python", "manage.py", "migrate"]
      restartPolicy: Never
```

```bash
# 创建 Job
kubectl apply -f job.yaml

# 快速创建
kubectl create job my-job --image=busybox -- echo "Hello from Job"

# 查看 Job
kubectl get jobs

# 查看 Job 的 Pod
kubectl get pods -l job-name=db-migration

# 查看日志
kubectl logs job/db-migration

# 删除 Job
kubectl delete job db-migration
```

### 11.2 CronJob（定时任务）

```yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: daily-backup
spec:
  schedule: "30 2 * * *"              # 每天凌晨 2:30
  concurrencyPolicy: Forbid           # 不允许并发执行
  successfulJobsHistoryLimit: 3       # 保留 3 个成功 Job
  failedJobsHistoryLimit: 1           # 保留 1 个失败 Job
  jobTemplate:
    spec:
      template:
        spec:
          containers:
            - name: backup
              image: myapp:1.0
              command: ["sh", "-c", "pg_dump -h db > /backup/db.sql"]
          restartPolicy: OnFailure
```

```bash
# 快速创建
kubectl create cronjob hourly-report --image=myapp:1.0 --schedule="0 * * * *" -- python report.py

# 查看 CronJob
kubectl get cronjobs
kubectl get cj

# 手动触发一次 CronJob
kubectl create job manual-backup --from=cronjob/daily-backup

# 暂停 CronJob
kubectl patch cronjob daily-backup -p '{"spec":{"suspend":true}}'

# 恢复 CronJob
kubectl patch cronjob daily-backup -p '{"spec":{"suspend":false}}'

# 删除 CronJob
kubectl delete cronjob daily-backup
```

---

## 12. 水平自动扩缩容（HPA）

```bash
# 基于 CPU 创建 HPA（CPU 使用率超过 50% 时扩容，2-10 个副本）
kubectl autoscale deployment my-app --min=2 --max=10 --cpu-percent=50

# 查看 HPA
kubectl get hpa

# 查看详情
kubectl describe hpa my-app

# 删除 HPA
kubectl delete hpa my-app
```

### HPA YAML 示例（基于多指标）

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: my-app-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: my-app
  minReplicas: 2
  maxReplicas: 20
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 50
    - type: Resource
      resource:
        name: memory
        target:
          type: Utilization
          averageUtilization: 70
```

---

## 13. 节点管理（Node）

```bash
# 查看节点
kubectl get nodes
kubectl get nodes -o wide

# 查看节点详情
kubectl describe node worker-01

# 查看节点资源使用（需 metrics-server）
kubectl top nodes

# 给节点打标签
kubectl label node worker-01 disktype=ssd

# 删除标签
kubectl label node worker-01 disktype-

# 给节点加污点（阻止 Pod 调度上来）
kubectl taint nodes worker-01 dedicated=gpu:NoSchedule

# 删除污点
kubectl taint nodes worker-01 dedicated=gpu:NoSchedule-

# 标记节点为不可调度（维护前）
kubectl cordon worker-01

# 驱逐节点上的 Pod（安全迁移）
kubectl drain worker-01 --ignore-daemonsets --delete-emptydir-data

# 恢复节点调度
kubectl uncordon worker-01
```

---

## 14. 标签和注解（Label & Annotation）

```bash
# ===== 标签 =====

# 添加标签
kubectl label pod my-pod env=production
kubectl label deployment my-app tier=frontend

# 覆盖已有标签
kubectl label pod my-pod env=staging --overwrite

# 删除标签
kubectl label pod my-pod env-

# 按标签查询
kubectl get pods -l env=production
kubectl get pods -l "env in (production,staging)"
kubectl get pods -l "env!=test"
kubectl get pods -l app=nginx,env=production

# 显示标签列
kubectl get pods --show-labels
kubectl get pods -L app,env    # 指定显示哪些标签

# ===== 注解 =====

# 添加注解
kubectl annotate pod my-pod description="Main application pod"

# 删除注解
kubectl annotate pod my-pod description-
```

---

## 15. 资源管理通用命令

### 15.1 apply / create / delete

```bash
# 声明式管理（推荐，可重复执行）
kubectl apply -f deployment.yaml
kubectl apply -f ./k8s/            # 应用整个目录
kubectl apply -f https://raw.githubusercontent.com/xxx/deployment.yaml    # 从 URL 应用

# 命令式创建（资源已存在会报错）
kubectl create -f deployment.yaml

# 删除资源
kubectl delete -f deployment.yaml
kubectl delete -f ./k8s/           # 删除整个目录定义的资源

# 批量操作
kubectl apply -f deployment.yaml -f service.yaml -f ingress.yaml

# 递归应用目录
kubectl apply -R -f ./k8s/
```

### 15.2 get / describe / explain

```bash
# 查看多种资源
kubectl get pods,svc,deploy

# 查看所有资源
kubectl get all
kubectl get all -n my-namespace

# 查看资源的 API 文档
kubectl explain pod
kubectl explain pod.spec.containers
kubectl explain deployment.spec.strategy --recursive

# 使用 JSONPath 提取信息
kubectl get pods -o jsonpath='{.items[*].spec.containers[*].image}'
kubectl get nodes -o jsonpath='{.items[*].status.addresses[?(@.type=="InternalIP")].address}'
```

### 15.3 edit / patch

```bash
# 使用编辑器修改资源
kubectl edit deployment my-app
KUBE_EDITOR="code --wait" kubectl edit deployment my-app    # 使用 VS Code

# JSON Patch
kubectl patch deployment my-app -p '{"spec":{"replicas":5}}'

# Strategic Merge Patch
kubectl patch deployment my-app --type=strategic -p '
spec:
  template:
    spec:
      containers:
      - name: app
        resources:
          limits:
            memory: "1Gi"
'

# JSON Patch（精确操作）
kubectl patch deployment my-app --type=json -p '[{"op":"replace","path":"/spec/replicas","value":5}]'
```

---

## 16. 调试与排错

### 16.1 常用排错流程

```bash
# 1. 查看 Pod 状态
kubectl get pods
# 常见异常状态：
# - Pending：等待调度（资源不足或节点选择器不匹配）
# - CrashLoopBackOff：容器反复崩溃
# - ImagePullBackOff：拉取镜像失败
# - ErrImagePull：镜像不存在或无权限
# - OOMKilled：内存超限被杀

# 2. 查看事件（按时间排序）
kubectl get events --sort-by=.metadata.creationTimestamp
kubectl get events -n my-namespace --field-selector type=Warning

# 3. 查看 Pod 详细信息
kubectl describe pod my-pod
# 重点看 Events 部分和 Conditions 部分

# 4. 查看日志
kubectl logs my-pod
kubectl logs my-pod --previous    # 崩溃前的日志

# 5. 进入容器排查
kubectl exec -it my-pod -- /bin/sh
```

### 16.2 debug 命令

```bash
# 创建调试容器（临时附加到 Pod）
kubectl debug my-pod -it --image=busybox

# 创建 Pod 的调试副本（修改镜像或命令）
kubectl debug my-pod -it --image=ubuntu --copy-to=debug-pod --share-processes

# 在节点上创建调试 Pod
kubectl debug node/worker-01 -it --image=ubuntu
```

### 16.3 DNS 排查

```bash
# 使用 dnsutils 容器测试 DNS
kubectl run dns-test --rm -it --image=busybox -- nslookup my-service
kubectl run dns-test --rm -it --image=busybox -- nslookup my-service.my-namespace.svc.cluster.local

# 查看 DNS 配置
kubectl exec my-pod -- cat /etc/resolv.conf
```

### 16.4 网络连通性排查

```bash
# 使用 netshoot 容器排查网络
kubectl run netshoot --rm -it --image=nicolaka/netshoot -- bash

# 在容器内可以使用：
# curl http://my-service:80
# ping my-service
# dig my-service.default.svc.cluster.local
# traceroute my-service
# tcpdump -i eth0
```

---

## 17. RBAC 权限管理

### 17.1 Role / ClusterRole

```yaml
# ===== 命名空间级别的 Role =====
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  namespace: dev
  name: pod-reader
rules:
  - apiGroups: [""]
    resources: ["pods", "pods/log"]
    verbs: ["get", "list", "watch"]
  - apiGroups: ["apps"]
    resources: ["deployments"]
    verbs: ["get", "list"]

---
# ===== 集群级别的 ClusterRole =====
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: cluster-viewer
rules:
  - apiGroups: [""]
    resources: ["nodes", "namespaces"]
    verbs: ["get", "list", "watch"]
```

### 17.2 RoleBinding / ClusterRoleBinding

```yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: read-pods
  namespace: dev
subjects:
  - kind: User
    name: alice
    apiGroup: rbac.authorization.k8s.io
  - kind: ServiceAccount
    name: my-sa
    namespace: dev
roleRef:
  kind: Role
  name: pod-reader
  apiGroup: rbac.authorization.k8s.io
```

### 17.3 RBAC 常用命令

```bash
# 创建 ServiceAccount
kubectl create serviceaccount my-sa -n dev

# 创建 Role
kubectl create role pod-reader --verb=get,list,watch --resource=pods -n dev

# 创建 RoleBinding
kubectl create rolebinding read-pods --role=pod-reader --serviceaccount=dev:my-sa -n dev

# 创建 ClusterRole
kubectl create clusterrole cluster-viewer --verb=get,list,watch --resource=nodes,namespaces

# 创建 ClusterRoleBinding
kubectl create clusterrolebinding view-all --clusterrole=cluster-viewer --user=alice

# 检查权限
kubectl auth can-i create pods
kubectl auth can-i delete deployments -n production
kubectl auth can-i "*" "*"    # 是否是管理员

# 以指定用户身份检查
kubectl auth can-i get pods --as alice -n dev
kubectl auth can-i get pods --as system:serviceaccount:dev:my-sa

# 查看角色和绑定
kubectl get roles -n dev
kubectl get rolebindings -n dev
kubectl get clusterroles
kubectl get clusterrolebindings
```

---

## 18. 实用技巧

### 18.1 kubectl 别名和自动补全

```bash
# 设置别名（加入 ~/.bashrc 或 ~/.zshrc）
alias k='kubectl'
alias kgp='kubectl get pods'
alias kgs='kubectl get svc'
alias kgd='kubectl get deploy'
alias kga='kubectl get all'
alias kaf='kubectl apply -f'
alias kdel='kubectl delete'
alias klog='kubectl logs -f'
alias kex='kubectl exec -it'

# 开启 bash 自动补全
source <(kubectl completion bash)
echo 'source <(kubectl completion bash)' >> ~/.bashrc
# 让别名也能补全
complete -o default -F __start_kubectl k

# zsh 自动补全
source <(kubectl completion zsh)
echo 'source <(kubectl completion zsh)' >> ~/.zshrc
```

### 18.2 常用的 dry-run 生成 YAML

```bash
# Pod
kubectl run my-pod --image=nginx --dry-run=client -o yaml > pod.yaml

# Deployment
kubectl create deployment my-app --image=myapp:1.0 --replicas=3 --dry-run=client -o yaml > deploy.yaml

# Service
kubectl expose deployment my-app --port=80 --target-port=8080 --dry-run=client -o yaml > svc.yaml

# Job
kubectl create job my-job --image=busybox --dry-run=client -o yaml -- echo hello > job.yaml

# CronJob
kubectl create cronjob my-cron --image=busybox --schedule="*/5 * * * *" --dry-run=client -o yaml -- echo hello > cron.yaml

# ConfigMap
kubectl create configmap my-config --from-literal=key=value --dry-run=client -o yaml > cm.yaml

# Secret
kubectl create secret generic my-secret --from-literal=password=secret --dry-run=client -o yaml > secret.yaml
```

### 18.3 快速查看资源间关系

```bash
# 查看 Deployment → ReplicaSet → Pod 的完整链路
kubectl get deploy,rs,pod -l app=my-app

# 查看 Service 对应的 Endpoints
kubectl get svc,endpoints -l app=my-app

# 查看某个命名空间下的所有资源
kubectl get all -n my-namespace

# 查看事件（排查调度和启动问题）
kubectl get events --sort-by=.lastTimestamp -n my-namespace
```

---

## 19. 命令速查表

### 基础操作

| 命令 | 功能 |
|------|------|
| `kubectl get <资源>` | 查看资源列表 |
| `kubectl describe <资源> <名称>` | 查看详细信息 |
| `kubectl create -f <文件>` | 命令式创建 |
| `kubectl apply -f <文件>` | 声明式创建/更新 |
| `kubectl delete <资源> <名称>` | 删除资源 |
| `kubectl edit <资源> <名称>` | 编辑资源 |
| `kubectl patch <资源> <名称> -p '{...}'` | 局部更新 |
| `kubectl explain <资源>` | 查看 API 文档 |

### Pod 操作

| 命令 | 功能 |
|------|------|
| `kubectl run <名称> --image=<镜像>` | 快速创建 Pod |
| `kubectl exec -it <Pod> -- bash` | 进入 Pod |
| `kubectl logs -f <Pod>` | 实时查看日志 |
| `kubectl cp <src> <Pod>:<dest>` | 复制文件 |
| `kubectl port-forward <Pod> 8080:80` | 端口转发 |
| `kubectl top pod` | 查看资源占用 |
| `kubectl debug <Pod> -it --image=busybox` | 调试 Pod |

### Deployment 操作

| 命令 | 功能 |
|------|------|
| `kubectl create deploy <名称> --image=<镜像>` | 创建 Deployment |
| `kubectl scale deploy <名称> --replicas=N` | 扩缩容 |
| `kubectl set image deploy/<名称> <容器>=<镜像>` | 更新镜像 |
| `kubectl rollout status deploy/<名称>` | 查看更新状态 |
| `kubectl rollout undo deploy/<名称>` | 回滚 |
| `kubectl rollout restart deploy/<名称>` | 重启所有 Pod |
| `kubectl rollout history deploy/<名称>` | 查看更新历史 |
| `kubectl autoscale deploy <名称> --min=2 --max=10` | 自动扩缩容 |

### 常用 get 参数

| 参数 | 功能 | 示例 |
|------|------|------|
| `-o wide` | 显示更多列 | `kubectl get pods -o wide` |
| `-o yaml` | YAML 格式输出 | `kubectl get pod my-pod -o yaml` |
| `-o json` | JSON 格式输出 | `kubectl get pod my-pod -o json` |
| `-o jsonpath` | 提取特定字段 | `kubectl get pods -o jsonpath='{.items[*].metadata.name}'` |
| `-l` | 按标签过滤 | `kubectl get pods -l app=nginx` |
| `-n` | 指定命名空间 | `kubectl get pods -n production` |
| `-A` | 所有命名空间 | `kubectl get pods -A` |
| `-w` | 实时监听 | `kubectl get pods -w` |
| `--sort-by` | 排序 | `kubectl get pods --sort-by=.metadata.creationTimestamp` |
| `--show-labels` | 显示标签 | `kubectl get pods --show-labels` |
| `--dry-run=client -o yaml` | 生成 YAML | `kubectl create deploy x --image=y --dry-run=client -o yaml` |
