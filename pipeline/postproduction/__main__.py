"""Run with python -m pipeline.postproduction."""

import argparse
import json
import subprocess

from .tools import plan_project, render_project, validate_project


def main():
    parser = argparse.ArgumentParser(description='AniFlow 独立视频后期剪辑')
    commands = parser.add_subparsers(dest='command', required=True)
    draft = commands.add_parser('plan', help='根据分镜和画面生成工程')
    draft.add_argument('storyboard')
    draft.add_argument('project')
    draft.add_argument('--width', type=int, default=1920)
    draft.add_argument('--height', type=int, default=1080)
    draft.add_argument('--fps', type=int, default=30)
    draft.add_argument('--vision-model')
    draft.add_argument('--replace', action='store_true')
    check = commands.add_parser('validate', help='校验已保存或手工编辑的工程')
    check.add_argument('project')
    export = commands.add_parser('render', help='渲染预览或最终 MP4 并报告画面检查结果')
    export.add_argument('project')
    export.add_argument('output')
    export.add_argument('--preview', action='store_true')
    args = parser.parse_args()
    operations = {'plan': plan_project, 'validate': validate_project, 'render': render_project}
    options = vars(args)
    operation = operations[options.pop('command')]
    try:
        result = operation(**options)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        parser.exit(1, f'{error}\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
