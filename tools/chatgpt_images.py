"""Open/reconnect AniFlow's Chrome, or process one image through its GPT chat UI."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pipeline.chatgpt_browser import ChatGPTBrowser
from pipeline.story_bible import StoryError


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['open','status','edit','delete-chat'])
    parser.add_argument('--input',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--watermark',action='store_true')
    parser.add_argument('--dialogue',action='store_true')
    parser.add_argument('--text',action='store_true')
    parser.add_argument('--feedback',help='针对残留水印或被改动区域的补充修正要求')
    parser.add_argument('--regions',type=Path,help='归一化水印矩形的JSON文件，先用纯色覆盖再交给模型修补')
    parser.add_argument('--checkpoint',type=Path,help='已完成任务的 .job.json')
    parser.add_argument('--confirm-delete',action='store_true',help='明确确认删除该任务的网页聊天记录')
    args=parser.parse_args()
    browser=ChatGPTBrowser()
    if args.command=='open':
        result=browser.open()
    elif args.command=='status':
        result=browser.check_login()
    elif args.command=='delete-chat':
        if not args.checkpoint:parser.error('delete-chat 需要 --checkpoint')
        result=browser.delete_job_chat(args.checkpoint,confirmed=args.confirm_delete)
    else:
        if not args.input or not args.output or not args.input.is_file():
            parser.error('edit 需要 --input 原图 和 --output 新PNG路径')
        if args.input.resolve()==args.output.resolve():
            parser.error('清图不能覆盖原图')
        args.output.parent.mkdir(parents=True,exist_ok=True)
        result=browser.edit_image(args.input.resolve(),args.output.resolve(),
            {k:getattr(args,k) for k in ('watermark','dialogue','text')},args.output.with_suffix('.job.json'),feedback=args.feedback,
            watermark_regions=json.loads(args.regions.read_text('utf-8-sig')) if args.regions else None)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    try:
        main()
    except StoryError as exc:
        print(str(exc),file=sys.stderr)
        sys.exit(1)
