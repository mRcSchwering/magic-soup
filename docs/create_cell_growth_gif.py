"""
Generate gif showing different cellmaps next to each other.

The cellmaps come from runs where cellmaps were logged as images using tensorboard.
They are pasted next to each other horizontally.

1. Create a run for every cellmap
2. Run this script with all run directories as arguments
3. Compress result gif using `gifsicle`
4. Check how it looks in documentation

```
PYTHONPATH=./python python docs/create_cell_growth_gif.py docs/runs/2023-12-21_10-31 docs/runs/2023-12-21_10-32 docs/runs/2023-12-21_10-33
gifsicle -i image.gif --optimize=3 --colors 32 -o image_o3_32.gif
bash scripts/serve-docs.sh
```
"""
import io
import contextlib
import argparse
from pathlib import Path
from PIL import Image, ImageOps
from tensorboard.backend.event_processing import event_accumulator


def create_concat_gif(
    eventdirs: list[str | Path],
    outfile: str | Path,
    image_tag: str,
    fps=10,
    loop=0,
    border=10,
):
    ea_kwargs = {"size_guidance": {event_accumulator.IMAGES: 0}}

    last_contents = []
    for eventdir in eventdirs:
        ea = event_accumulator.EventAccumulator(str(eventdir), **ea_kwargs)
        ea.Reload()
        *_, last = iter(ea.Images(image_tag))
        last_contents.append(io.BytesIO(last.encoded_image_string))

    eas = []
    for eventdir in eventdirs:
        ea = event_accumulator.EventAccumulator(str(eventdir), **ea_kwargs)
        ea.Reload()
        eas.append(iter(ea.Images(image_tag)))

    content_lsts = []
    is_empty = [False] * len(eas)
    while not all(is_empty):
        content_lst = []
        for ea_i, ea in enumerate(eas):
            try:
                obj = next(ea)
                content_lst.append(io.BytesIO(obj.encoded_image_string))
            except StopIteration:
                content_lst.append(last_contents[ea_i])
                is_empty[ea_i] = True
        content_lsts.append(content_lst)

    with contextlib.ExitStack() as stack:
        imgs = []
        for content_lst in content_lsts:
            img_lst = []
            for content in content_lst:
                img = stack.enter_context(Image.open(content))
                img = ImageOps.expand(image=img, border=border, fill="white")
                img_lst.append(img)

            widths, heights = zip(*(d.size for d in img_lst))
            new_img = Image.new("RGB", (sum(widths), max(heights)))
            x_offset = 0
            for img in img_lst:
                new_img.paste(img, (x_offset, 0))
                x_offset += img.size[0]

            imgs.append(new_img)

        img = imgs[0]
        img.save(
            fp=str(outfile),
            format="GIF",
            append_images=imgs,
            save_all=True,
            duration=1000 / fps,
            loop=loop,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("eventdirs", nargs="+", type=str)
    parser.add_argument("--outfile", default="example/image.gif", type=str)
    parser.add_argument("--image-tag", default="Maps/Cells", type=str)
    parser.add_argument("--fps", default=10, type=float)
    parser.add_argument("--border", default=10, type=int)
    args = parser.parse_args()

    create_concat_gif(
        eventdirs=args.eventdirs,
        outfile=args.outfile,
        image_tag=args.image_tag,
        fps=args.fps,
        border=args.border,
    )
