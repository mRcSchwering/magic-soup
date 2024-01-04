# type: ignore
"""
Generate cover gif from a successful run

Loads molecule concentrations, cell count, and cell map over time
and for each step creates a timeseries plot for molecule concentrations and cell count,
and multiple scatter plots for displaying cell maps.
The cell maps show all cells, and the most abundant celllines only at a certain step (`--top-step`).

1. Create a good example run with many saves
2. Choose step `--top-step` where celllines should be counted (e.g. at cell count peak)
3. Run this script with low `--to-step` and check output
4. See parameters of `main()` for adjusting output image (crop, legend alignment, layout, size)
5. Run for all steps
6. Compress result gif using `gifsicle`
7. Check how rendered gif looks in documentation
8. (PyPI needs <5Mb, cut/compress further for PyPI)

```
PYTHONPATH=./python python docs/create_cover_gif.py docs/runs/2023-12-21_10-31 --top-step 1000
gifsicle -i image.gif --optimize=3 --colors 32 -o animation.gif
bash scripts/serve-docs.sh

# optionally cut gif length
gifsicle -U animation.gif `seq -f "#%g" 0 1 700` --optimize=3 -o 'animation[small].gif'
```
"""
from pathlib import Path
from collections import Counter
import argparse
import io
import pandas as pd
from PIL import Image
from plotnine import *
import magicsoup as ms

theme_set(theme_minimal())


def _collect_scalars(
    rundir: Path, steps: list[int], world: ms.World
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cell_records = []
    mol_records = []
    for step in steps:
        world.load_state(statedir=rundir / f"step={step}", ignore_cell_params=True)
        cell_records.append({"step": step, "cells": world.n_cells})
        for mol in world.chemistry.molecules:
            mol_i = world.chemistry.mol_2_idx[mol]
            mol_records.append(
                {
                    "step": step,
                    "m": mol.name,
                    "avg": world.molecule_map[mol_i].mean().item(),
                }
            )

    cells_df = pd.DataFrame.from_records(cell_records)
    mols_df = pd.DataFrame.from_records(mol_records)
    return cells_df, mols_df


def _collect_cellmaps(
    rundir: Path, steps: list[int], world: ms.World, top_step: int, top_n=3
) -> pd.DataFrame:
    world.load_state(statedir=rundir / f"step={top_step}", ignore_cell_params=True)
    top_labels = {
        f"CL{i}": d[0]
        for i, d in enumerate(Counter(world.cell_labels).most_common(n=top_n))
    }

    cellmap_records = []
    for step in steps:
        world.load_state(statedir=rundir / f"step={step}", ignore_cell_params=True)
        for cell_i in range(world.n_cells):
            x, y = world.cell_positions[cell_i].tolist()
            label = world.cell_labels[cell_i]
            cellmap_records.append({"step": step, "x": x, "y": y, "label": label})

    all_df = pd.DataFrame.from_records(cellmap_records)
    all_df["grp"] = "all"

    dfs = []
    for name, label in top_labels.items():
        df = all_df[all_df["label"] == label].copy()
        df["grp"] = name
        dfs.append(df)

    df = pd.concat([all_df] + dfs, ignore_index=True)
    df["grp"] = pd.Categorical(df["grp"], ["all"] + sorted(top_labels))
    return df


def _plot_timeseries(
    mols_df: pd.DataFrame,
    cells_df: pd.DataFrame,
    color_map: dict | None = None,
    alpha_map: dict | None = None,
    size_map: dict | None = None,
    text_offset_map: dict | None = None,
    marker_offset_map: dict | None = None,
    vline: float | None = None,
    marker_len=0.15,
    legend_offset=0.02,
    figsize=(14, 5),
    marker_size=2,
    text_size=10,
) -> ggplot:
    if color_map is None:
        color_map = {"cells": "#9494DE", "energy": "#FFB34D", "metabolite": "#747474"}
    if alpha_map is None:
        alpha_map = {"cells": 1.0, "energy": 1.0, "metabolite": 0.5}
    if size_map is None:
        size_map = {"cells": 1.0, "energy": 1.0, "metabolite": 0.5}
    if text_offset_map is None:
        text_offset_map = {"cells": 0.6, "energy": 0.6}
    if marker_offset_map is None:
        marker_offset_map = {"cells": 0.3, "energy": 0.25}

    mols_df["g"] = "metabolite"
    mols_df.loc[mols_df["m"] == "ATP", "g"] = "energy"
    mols_df["g"] = pd.Categorical(mols_df["g"], ["energy", "metabolite"])
    mols_df["df"] = "B"
    cells_df["g"] = "cells"
    cells_df["df"] = "A"

    legend_x = cells_df["step"].max() * (1 + legend_offset)
    ymax_cells = cells_df["cells"].max()
    ymax_mols = mols_df["avg"].max()

    start_cells = marker_offset_map["cells"] * ymax_cells
    end_cells = start_cells + marker_len * ymax_cells
    text_cells = text_offset_map["cells"] * ymax_cells

    start_energy = marker_offset_map["energy"] * ymax_mols
    end_energy = start_energy + marker_len * ymax_mols
    text_energy = text_offset_map["energy"] * ymax_mols

    legend_lines_df = pd.DataFrame.from_records(
        [
            {"x": legend_x, "y": start_cells, "df": "A", "c": "cells"},
            {"x": legend_x, "y": end_cells, "df": "A", "c": "cells"},
            {"x": legend_x, "y": start_energy, "df": "B", "c": "energy"},
            {"x": legend_x, "y": end_energy, "df": "B", "c": "energy"},
        ]
    )

    legend_text_df = pd.DataFrame.from_records(
        [
            {"x": legend_x, "y": text_cells, "df": "A", "l": "cells"},
            {"x": legend_x, "y": text_energy, "df": "B", "l": "energy"},
        ]
    )

    # fmt: off
    g = (
        ggplot()
        + geom_area(aes(x="step", y="cells", fill="g"), data=cells_df)
        + geom_line(aes(x="step", y="avg", color="g", size="g", group="m", alpha="g"), data=mols_df)
        + geom_line(aes(x="x", y="y", color="c"), size=marker_size, data=legend_lines_df)
        + geom_text(aes(x="x", y="y", label="l"), size=text_size, angle=-90, data=legend_text_df)
        + facet_grid("df ~ .", scales="free_y")
        + scale_fill_manual(color_map, drop=False)
        + scale_color_manual(color_map, drop=False)
        + scale_alpha_manual(alpha_map, drop=False)
        + scale_size_manual(size_map, drop=False)
        + theme(strip_background=element_blank(), strip_text_y=element_blank())
        + theme(panel_grid_major=element_blank(), panel_grid_minor=element_blank())
        + theme(legend_position="none")
        + theme(axis_title=element_blank(), axis_text=element_blank())
        + theme(figure_size=figsize)
    )
    # fmt: on
    if vline is not None:
        g = g + geom_vline(xintercept=vline, linetype="dashed", color="gray")
    return g


def _plot_cellmaps(
    df: pd.DataFrame,
    map_size: int,
    color_map: dict | None = None,
    text_pad: dict | None = None,
    figsize=(8, 2),
    text_size=10,
) -> ggplot:
    if text_pad is None:
        text_pad = {"x": 0.05, "y": 0.03}
    if color_map is None:
        color_map = {
            "all": "dimgray",
            "CL0": "#35AF7B",
            "CL1": "#3976A4",
            "CL2": "#8860C4",
        }

    textx = map_size * (1 - text_pad["x"])
    texty = map_size * (1 - text_pad["y"])
    text_df = pd.DataFrame.from_records(
        [{"x": textx, "y": texty, "grp": d} for d in color_map.keys()]
    )
    text_df["grp"] = pd.Categorical(text_df["grp"], color_map)

    # fmt: off
    g = (ggplot(df, aes(x="x", y="y"))
        + geom_point(aes(color="grp"), size=.1)
        + coord_fixed(ratio=1, xlim=(0, map_size), ylim=(0, map_size))
        + scale_color_manual(color_map, drop=False)
        + geom_text(aes(label="grp"), size=text_size, data=text_df)
        + facet_grid(". ~ grp")
        + theme(legend_position="none")
        + theme(strip_background=element_blank(), strip_text=element_blank())
        + theme(plot_margin=0, panel_spacing=0)
        + theme(panel_background=element_blank(), panel_border=element_rect(colour="black", size=0.5))
        + theme(panel_grid_major=element_blank(), panel_grid_minor=element_blank())
        + theme(axis_title=element_blank(), axis_text=element_blank())
        + theme(figure_size=figsize))
    # fmt: on
    return g


def _ggplot_2_pil(plot: ggplot, width=8, height=5, dpi=200) -> Image:
    buf = io.BytesIO()
    plot.save(buf, width=width, height=height, dpi=dpi)
    buf.seek(0)
    return Image.open(buf)


def _crop_img(img: Image, pad: tuple[int, ...]) -> Image:
    w, h = img.size
    if len(pad) == 1:
        lft = pad[0]
        up = pad[0]
        rgt = w - pad[0]
        lo = h - pad[0]
    elif len(pad) == 2:
        lft = pad[0]
        up = pad[1]
        rgt = w - pad[0]
        lo = h - pad[1]
    elif len(pad) == 4:
        lft = pad[0]
        up = pad[1]
        rgt = w - pad[2]
        lo = h - pad[3]
    return img.crop((lft, up, rgt, lo))


def main(
    rundir: Path,
    outfile: Path,
    top_step: int,
    to_step=-1,
    ts_width=11,
    ts_height=3,
    cm_width=8,
    cm_height=2,
    dpi=200,
    ts_crop=(50, 10),
    cm_crop=(10, 10),
    fps=10,
):
    world = ms.World.from_file(rundir=rundir, device="cpu")
    steps = sorted(int(d.name.split("step=")[1]) for d in rundir.glob("step=*"))
    if to_step == -1:
        to_step = steps[-1]

    cells_df, mols_df = _collect_scalars(rundir=rundir, steps=steps, world=world)
    cellmaps_df = _collect_cellmaps(
        rundir=rundir, steps=steps, world=world, top_step=top_step
    )

    imgs = []
    for step in steps:
        if step > to_step:
            break

        ts_plot = _plot_timeseries(
            marker_len=0.2,
            size_map={"cells": 1, "energy": 1, "metabolite": 0.3},
            text_offset_map={"cells": 0.7, "energy": 0.7},
            cells_df=cells_df,
            mols_df=mols_df,
            vline=step,
            figsize=(ts_width, ts_height),
        )
        cm_plot = _plot_cellmaps(
            df=cellmaps_df[cellmaps_df["step"] == step],
            map_size=world.map_size,
            figsize=(cm_width, cm_height),
        )

        ts_img = _ggplot_2_pil(plot=ts_plot, width=ts_width, height=ts_height, dpi=dpi)
        cm_img = _ggplot_2_pil(plot=cm_plot, width=cm_width, height=cm_height, dpi=dpi)
        ts_img = _crop_img(ts_img, ts_crop)
        cm_img = _crop_img(cm_img, cm_crop)
        ts_w, ts_h = ts_img.size
        cm_w, cm_h = cm_img.size

        target_w = max(ts_w, cm_w)
        target_h = ts_h + cm_h
        bkg = Image.new("RGBA", (target_w, target_h), (255, 255, 255, 255))
        bkg_w, _ = bkg.size

        cm_offset = ((bkg_w - cm_w) // 2, 0)
        ts_offset = ((bkg_w - ts_w) // 2, cm_h)
        bkg.paste(cm_img, cm_offset, cm_img)
        bkg.paste(ts_img, ts_offset, ts_img)
        imgs.append(bkg)

    img = imgs[0]
    img.save(
        fp=str(outfile),
        format="GIF",
        append_images=imgs,
        save_all=True,
        duration=1000 / fps,
        loop=0,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("rundir", type=str)
    parser.add_argument("--top-step", default=1000, type=int)
    parser.add_argument("--to-step", default=-1, type=int)
    parser.add_argument("--outfile", default="example/image.gif", type=str)
    parser.add_argument("--fps", default=10, type=float)
    args = parser.parse_args()
    main(
        rundir=Path(args.rundir),
        outfile=Path(args.outfile),
        to_step=args.to_step,
        top_step=args.top_step,
        fps=args.fps,
    )
