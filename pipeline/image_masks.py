"""Deterministic solid-colour input masks; immutable source images are untouched."""
import math
from pathlib import Path

from PIL import Image, ImageDraw

from pipeline.story_bible import StoryError

MASK_COLOR = '#ff00ff'


def mask_regions(value):
    if value is None:
        return []
    if not isinstance(value,list) or len(value)>20:
        raise StoryError('水印遮盖区域需为最多20个矩形')
    result=[]
    for box in value:
        if (not isinstance(box,list) or len(box)!=4 or
                any(type(v) not in (int,float) or not math.isfinite(v) for v in box)):
            raise StoryError('遮盖矩形需使用归一化 [x,y,宽,高]')
        x,y,w,h=box
        if x<0 or y<0 or w<=0 or h<=0 or x+w>1.000001 or y+h>1.000001:
            raise StoryError('遮盖区域超出原图')
        result.append([float(v) for v in box])
    return result


def solid_mask(original,destination,regions):
    original,destination=Path(original).resolve(),Path(destination).resolve()
    if original==destination:
        raise StoryError('遮盖副本不能覆盖原图')
    regions=mask_regions(regions)
    if not regions:
        raise StoryError('请先标记水印区域')
    with Image.open(original) as opened:
        image=opened.convert('RGB')
    draw=ImageDraw.Draw(image)
    for x,y,w,h in regions:
        left,top=math.floor(x*image.width),math.floor(y*image.height)
        right,bottom=min(image.width,math.ceil((x+w)*image.width)),min(image.height,math.ceil((y+h)*image.height))
        if right<=left or bottom<=top:
            raise StoryError('遮盖区域小于一个像素')
        draw.rectangle((left,top,right-1,bottom-1),fill=MASK_COLOR)
    destination.parent.mkdir(parents=True,exist_ok=True)
    image.save(destination,format='PNG')
    return destination
