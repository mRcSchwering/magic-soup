from pathlib import Path

import magicsoup as ms
import pandas as pd
from magicsoup.biology import CatalyticDomain, RegulatoryDomain, TransporterDomain
from magicsoup.examples.wood_ljungdahl import CHEMISTRY
from plotnine import *  # pylint: disable=W0614,W0401


def _plot_genome_transcripts(cell_idx: int, world: ms.World, w=14, h=0.2, gw=5, cdsw=4):
    cell = world.get_cell(by_idx=cell_idx)
    n = len(cell.genome)

    colors = {
        "genome": "dimgray",
        "CDS": "lightgray",
        "catalytic": "#fe218b",
        "transporter": "#21b0fe",
        "regulatory": "#fed700",
    }
    dom_type_map = {
        CatalyticDomain: "catalytic",
        TransporterDomain: "transporter",
        RegulatoryDomain: "regulatory",
    }
    records = [{"tag": "genome", "dir": "", "start": 0, "stop": n, "type": "genome"}]
    for pi, prot in enumerate(cell.proteome):
        tag = f"CDS{pi}"
        start = prot.cds_start if prot.is_53 else n - prot.cds_start
        stop = prot.cds_end if prot.is_53 else n - prot.cds_end
        records.append(
            {
                "tag": tag,
                "dir": "fwd" if prot.is_53 else "bwd",
                "start": start,
                "stop": stop,
                "type": "CDS",
            }
        )
        for dom in prot.domains:
            records.append(
                {
                    "tag": tag,
                    "dir": "fwd" if prot.is_53 else "bwd",
                    "start": start + dom.start if prot.is_53 else start - dom.start,
                    "stop": start + dom.end if prot.is_53 else start - dom.end,
                    "type": dom_type_map[type(dom)],
                }
            )
    df = pd.DataFrame.from_records(records)

    tags = (
        df.loc[df["dir"] == "fwd", "tag"].unique().tolist()
        + ["genome"]
        + df.loc[df["dir"] == "bwd", "tag"].unique().tolist()
    )
    df["tag"] = pd.Categorical(df["tag"], categories=reversed(tags), ordered=True)
    df["type"] = pd.Categorical(df["type"], categories=list(colors), ordered=True)

    sizes = {d: gw if d == "genome" else cdsw for d in tags}
    return (
        ggplot(df)
        + geom_segment(
            aes(x="start", y="tag", xend="stop", yend="tag", color="type", size="tag")
        )
        + scale_color_manual(values=colors)  # pylint: disable=all
        + scale_size_manual(values=sizes)
        + guides(size=False)
        + theme(figure_size=(w, h * len(sizes) + 0.3))
        + theme(panel_grid_major=element_blank(), panel_grid_minor=element_blank())
        + theme(axis_title_x=element_blank(), axis_title_y=element_blank())
        + theme(
            legend_position="bottom", legend_title=element_blank(), legend_margin=10.0
        )
    ), "transcriptome"


def create_plots(imgs_dir: Path):
    world = ms.World(chemistry=CHEMISTRY)
    world.spawn_cells(genomes=[ms.random_genome(s=1000) for _ in range(10)])

    for idx in range(3):
        g, name = _plot_genome_transcripts(cell_idx=idx, world=world)
        g.save(imgs_dir / f"{name}{idx}.png", dpi=200, transparent=True)
