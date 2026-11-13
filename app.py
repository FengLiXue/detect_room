from __future__ import annotations

import argparse
import json

from src.annotation.door_annotator import annotate_video
from src.config.loader import load_json
from src.pipeline.runner import run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="多房间门口人数统计原型")
    parser.add_argument("--video", help="输入本地视频")
    parser.add_argument("--config", default="config/doors.json", help="门口配置 JSON")
    parser.add_argument("--model", default="yolo11n-pose.pt", help="姿态模型权重，默认 yolo11n-pose.pt")
    parser.add_argument("--output", default="outputs/annotated.mp4", help="可视化视频输出路径；使用 --no-render 禁用")
    parser.add_argument("--events", default="outputs/events.jsonl", help="事件 JSONL 输出路径")
    parser.add_argument("--fps", type=float, default=15.0, help="模拟在线读取帧率")
    parser.add_argument("--conf", type=float, default=0.35, help="人体检测置信度阈值")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO 输入尺寸")
    parser.add_argument("--device", default=None, help="如 cpu、0")
    parser.add_argument("--no-realtime", action="store_true", help="不等待，尽快处理离线视频")
    parser.add_argument("--no-render", action="store_true", help="不输出带标注视频")
    parser.add_argument("--display", action="store_true", help="显示实时预览窗口")
    parser.add_argument("--annotate", action="store_true", help="打开首帧门口标注工具，不运行检测")
    parser.add_argument("--room-prefix", default="room", help="标注时房间默认名称前缀")
    parser.add_argument("--single-door", action="store_true", help="标注完成一个门后退出")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.video:
        raise SystemExit("请提供 --video；标注模式示例：python app.py --annotate --video data/input.mp4")

    if args.annotate:
        annotate_video(args.video, args.config, room_prefix=args.room_prefix, single_door=args.single_door)
        return

    result = run(
        video_path=args.video,
        config=load_json(args.config),
        model_path=args.model,
        output_path=None if args.no_render else args.output,
        events_path=args.events,
        fps=args.fps,
        confidence=args.conf,
        image_size=args.imgsz,
        device=args.device,
        realtime=not args.no_realtime,
        display=args.display,
    )
    print(json.dumps({"final_counts": result.counts}, ensure_ascii=False))


if __name__ == "__main__":
    main()
