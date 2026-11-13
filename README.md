# Doorway Occupancy Detector

这是一个以离线视频模拟在线输入的模块化原型，用摄像头画面中的门槛线和门框线统计房间进出人数。

当前版本：

- 人体姿态：Ultralytics YOLO11n-pose（可切换为 `yolo11s-pose.pt`）
- 多目标跟踪：ByteTrack
- 输入：本地视频文件，按 15 FPS 节奏读取
- 门口标注：可拖动四个角的门洞四边形
- 输出：带可视化的视频和逐事件 JSONL 日志

## 安装

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

首次运行时 Ultralytics 会按模型名称下载权重。也可以先把权重文件放在本地，然后通过 `--model` 指定路径。

## 运行

```bash
python app.py \
  --video data/input.mp4 \
  --config config/doors.json \
  --model yolo11n-pose.pt \
  --output outputs/annotated.mp4 \
  --events outputs/events.jsonl \
  --fps 15
```

使用更大的姿态模型：

```bash
python app.py --video data/input.mp4 --config config/doors.json --model yolo11s-pose.pt
```

先不输出视频，只检查事件：

```bash
python app.py --video data/input.mp4 --config config/doors.json --no-render
```

## 门口配置

`config/doors.json` 中的坐标是相对于原始视频宽高的归一化坐标，范围为 `[0, 1]`。每个门配置：

- `polygon`：门洞四边形的四个角，顺序沿边界绕一圈；
- `room_id`、`room_name`：房间标识。

示例配置中的坐标仅用于演示，实际使用时需要根据视频画面重新标注。

## 模块结构

```text
src/
  common/       数据结构和通用类型
  config/       配置加载和校验
  geometry/     门洞四边形
  input/        离线视频模拟实时输入
  models/       YOLO 检测器适配
  tracking/     ByteTrack 实现
  events/       基于轨迹的进出状态机
  output/       视频绘制和事件日志
  pipeline/     端到端运行编排
```

## 设计限制

Track ID 是单摄像头内的临时轨迹编号，不是人的身份。每条轨迹按「门外、靠近、头和脚踝进入门洞、轨迹消失」记一次进入，进入后这条轨迹的状态清掉。脚踝没检到时只看头；检到了，脚踝也必须到门口。脚明显低于下沿时，人在镜头和门之间，身体挡住门洞也不计数。进门的人离开画面后，下一次有人出现在这扇门口，记出来。

## 交互式门口标注

项目提供首帧标注工具，不需要手写 `doors.json`。准备一段能看到门口的本地视频后运行：

```bash
python app.py \
  --annotate \
  --video data/input.mp4 \
  --config config/doors.json \
  --room-prefix room
```

窗口里先放一个四边形。拖动四个角，或按住框内空白移动整框，让它贴住门洞。按 `Enter` 后在终端输入房间名称和初始人数。人数是整数，从初始人数开始累计，不设置容量。直接回车则初始人数为 0。接着拖下一扇门。`R` 把当前四边形恢复成默认大小，`S` 保存已确认的门并退出，`Q` 或 `Esc` 退出且不保存。坐标会自动保存为相对于视频宽高的归一化坐标。

默认模型是 `yolo11n-pose.pt`，首次运行时由 Ultralytics 下载。远门上的人较小，可以改用 `--model yolo11s-pose.pt`。
