# Skill Hub — SOPHON TPU 模型部署技能库

帮助客户将 AI 模型部署到算能（SOPHON）BM1684X TPU 设备上的技能合集。

## 在 Claude Code 中使用

将本仓库的 `skills/` 目录软链接或复制到 `~/.claude/skills/`，Claude Code 即可自动加载，之后在对话中直接描述任务，Claude 会自动调用对应 skill：

```bash
git clone https://github.com/baifengbai/skill_hub.git
ln -s "$(pwd)/skill_hub/skills/funasr-bmodel" ~/.claude/skills/funasr-bmodel
ln -s "$(pwd)/skill_hub/skills/yolov5-bmodel" ~/.claude/skills/yolov5-bmodel
ln -s "$(pwd)/skill_hub/skills/ppocr-bmodel"  ~/.claude/skills/ppocr-bmodel
```

加载后，在 Claude Code 中说「帮我在 BM1684X 设备上部署 FunASR」即可自动调用对应 skill。

每个 skill 包含：
- 完整的部署步骤与踩坑记录
- Flask Web 前端（麦克风录音 / 文件上传）
- REST API 接口说明
- 常见问题排查

## 设备环境

| 项目 | 说明 |
|------|------|
| 芯片 | SOPHON BM1684X SOC |
| TPU 内存 | 13.5 GB |
| 系统内存 | 1.5 GB（需配置 swap） |
| 系统 | Ubuntu 20.04 aarch64 |
| 推理引擎 | tpu_perf / sophon.sail |
| Python | `/data/AIGC-SDK/hub_venv`（Python 3.10） |

## Skill 列表

| Skill | 模型 | 说明 |
|-------|------|------|
| [funasr-bmodel](./skills/funasr-bmodel/) | Paraformer Large | 中文语音识别（ASR + VAD + 标点） |
| [yolov5-bmodel](./skills/yolov5-bmodel/) | YOLOv5s | COCO 80 类目标检测，含可调阈值 |
| [ppocr-bmodel](./skills/ppocr-bmodel/) | PP-OCRv4 | 中文文字检测与识别（两阶段） |
| [bytetrack-bmodel](./skills/bytetrack-bmodel/) | YOLOv5s + ByteTrack | 多目标追踪，自动选择 bmcv/TPU 或 OpenCV/CPU 后端 |
| [vila-bmodel](./skills/vila-bmodel/) | VILA-1.5-3B | 视觉语言模型图片/视频问答，SSE 流式输出，不依赖 transformers |
| [qwen3-bmodel](./skills/qwen3-bmodel/) | Qwen3-4B AWQ W4BF16 | 对话 LLM，SSE 流式输出，单轮模式（seq512 限制）|
| [qwen3_5-bmodel](./skills/qwen3_5-bmodel/) | Qwen3.5-VL-2B int4 W4BF16 | 多模态 VL（图片/视频/文字问答），SSE 流式，seq2048 |

## 为什么用 gssh 而不是 SSH

本库所有远程操作均通过 [gssh](https://www.npmjs.com/package/@zzzwy/gssh) 完成，而非原生 SSH。原因如下：

| 对比项 | 原生 SSH | gssh |
|--------|---------|------|
| **Agent 工作焦点** | `ssh user@host` 会把 Agent 的整个执行环境切换到远端 shell，本地上下文丢失，后续操作极易出错 | 所有命令通过 `gssh exec "..."` 调用，Agent 始终在本地运行，远端操作只是一次普通函数调用 |
| **会话管理** | 每条命令都要重新建立连接，或依赖 ControlMaster 保持长连接，配置复杂 | daemon 常驻后台，session 自动复用，断线后 5 秒内自动重连 |
| **端口转发** | 需要额外的 `-L`/`-R` 参数，且与主连接绑定，主连接断则转发断 | `gssh forward -l 本地端口 -r 远端端口` 独立管理，随时增删，不受 exec 影响 |
| **文件传输** | 需要切换到 scp/rsync 命令，与 SSH session 分离 | `gssh scp` / `gssh sync` 与同一 session 复用，无需额外认证 |
| **sudo 支持** | 交互式密码输入，脚本化困难 | `gssh exec --sudo --sudo-password "xxx" "command"` 完全非交互 |
| **超时控制** | 长命令挂起会卡住整个 shell | `gssh exec -t 30 "command"` 超时后远端进程被 SIGKILL，本地立即恢复 |

**核心原则：Agent 的工作焦点永远在本地。** gssh 让远端设备操作变成本地的一个工具调用，而不是把 Agent "传送"到远端去。

### 安装与启动

```bash
npm install -g @zzzwy/gssh
gssh-daemon &   # 或用 pm2 管理：pm2 start gssh-daemon --name gssh
```

### 连接设备

```bash
gssh connect -u <USERNAME> -h <DEVICE_IP> -p <SSH_PORT> -P "<PASSWORD>"
```

## 快速开始

1. 克隆本仓库到本地
2. 进入对应 skill 目录，按照 `SKILL.md` 逐步操作
3. 用 gssh 连接设备并执行部署（见上方说明）
4. 通过 `gssh forward` 将服务映射到本地浏览器访问

## 通用前置条件

```bash
# 1. 在 /data 创建 2GB swap（系统内存仅 1.5GB，必须）
fallocate -l 2G /data/swapfile && chmod 600 /data/swapfile
mkswap /data/swapfile && swapon /data/swapfile
echo '/data/swapfile none swap sw 0 0' >> /etc/fstab

# 2. 安装 ffmpeg
apt-get install -y ffmpeg

# 3. 使用 hub_venv（Python 3.10，torch 已针对该平台编译）
/data/AIGC-SDK/hub_venv/bin/python3 your_service.py
```

> ⚠️ **不要用系统 Python 3.8 + PyPI torch**：会报 `Illegal instruction`，因指令集不兼容。
