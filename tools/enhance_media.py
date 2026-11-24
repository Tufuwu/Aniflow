"""Run with python -m tools.enhance_media."""
import argparse
import json

from pipeline.quality_enhancement import EnhancementOptions, EnhancementService
from pipeline.quality_models import hardware


def main():
    parser = argparse.ArgumentParser(description="AniFlow 本地画质增强")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("hardware")
    images = commands.add_parser("images")
    images.add_argument("sources", nargs="+")
    images.add_argument("--target", type=int, nargs=2, metavar=("WIDTH", "HEIGHT"), required=True)
    images.add_argument("--zoom", type=float, default=1.)
    images.add_argument("--face", action="store_true")
    images.add_argument("--fidelity", type=float, default=.9)
    video = commands.add_parser("video")
    video.add_argument("source")
    video.add_argument("destination")
    video.add_argument("--scale", type=int, choices=[2, 4], default=2)
    video.add_argument("--temporal", type=float, default=.45)
    for cmd in (images, video):
        cmd.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
        cmd.add_argument("--tile", type=int, default=0)
    args = parser.parse_args()
    if args.command == "hardware":
        result = hardware()
    else:
        options = EnhancementOptions.parse({"enabled": True, "device": args.device, "tile": args.tile,
                                            "face": getattr(args, "face", False),
                                            "fidelity": getattr(args, "fidelity", .9)})
        if args.command == "images":
            result = EnhancementService().batch(args.sources, args.target, options, zoom=args.zoom)
        else:
            from pipeline.quality_video import enhance_video
            result = enhance_video(args.source, args.destination, args.scale, options, args.temporal)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
