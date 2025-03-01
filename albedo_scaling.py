from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Type

from rfstudio.engine.task import Task, TaskGroup
from rfstudio.graphics import Cameras, Texture2D, TextureCubeMap, TriangleMesh, RGBAImages, PBRAImages, RGBImages
from rfstudio.graphics.shaders import BaseShader, ShadingContext, NormalShader
from rfstudio.io import dump_float32_image, load_float32_image
from rfstudio.data import RelightDataset
from rfstudio.visualization import TabularFigures
from rfstudio.ui import console
from rfstudio.utils.pretty import P

from jaxtyping import Float32
from torch import Tensor
import torch
import numpy as np


@dataclass
class Tester(Task):

    material: Path = ...
    dataset: RelightDataset = RelightDataset(path=...)

    @torch.no_grad()
    def run(self) -> None:

        self.dataset.to(self.device)
        cameras = self.dataset.get_inputs(split='test')[...]
        gt_albedos, *_ = self.dataset.get_meta(split='test')

        with console.progress(desc='Compute Albedo Scaling') as ptrack:
            albedo_scalings = []
            for i, camera in enumerate(ptrack(cameras)):
                mask = gt_albedos[i].item()[..., 3:]
                pred_albedo = RGBImages([load_float32_image(self.material / f'val_{i:06d}_kd.png')]).to(self.device).resize_to(800, 800).srgb2rgb().item() * mask
                gt_albedo = gt_albedos[i].blend((0, 0, 0)).srgb2rgb().item()
                albedo_scalings.append((pred_albedo * gt_albedo).view(-1, 3).sum(0) / pred_albedo.view(-1, 3).square().sum(0))

        albedo_scaling = torch.stack(albedo_scalings).mean(0)
        r, g, b = albedo_scaling.tolist()
        console.print(P@'Albedo Scaling: {r}, {g}, {b}')
        np.save(self.material.parent / 'mesh' / 'albedo_scaling.npy', albedo_scaling.cpu().numpy())

if __name__ == '__main__':
    TaskGroup(
        s4r_air=Tester(
            material=Path('out') / 's4r_air' / 'validate',
            dataset=RelightDataset(path=Path('..') / 'RadianceFieldStudio' / 'data' / 'Synthetic4Relight' / 'air_baloons'),
            cuda=0,
        ),
        s4r_chair=Tester(
            material=Path('out') / 's4r_chair' / 'validate',
            dataset=RelightDataset(path=Path('..') / 'RadianceFieldStudio' / 'data' / 'Synthetic4Relight' / 'chair'),
            cuda=0,
        ),
        s4r_hotdog=Tester(
            material=Path('out') / 's4r_hotdog' / 'validate',
            dataset=RelightDataset(path=Path('..') / 'RadianceFieldStudio' / 'data' / 'Synthetic4Relight' / 'hotdog'),
            cuda=0,
        ),
        s4r_jugs=Tester(
            material=Path('out') / 's4r_jugs' / 'validate',
            dataset=RelightDataset(path=Path('..') / 'RadianceFieldStudio' / 'data' / 'Synthetic4Relight' / 'jugs'),
            cuda=0,
        ),
        tsir_lego=Tester(
            material=Path('out') / 'tsir_lego' / 'validate',
            dataset=RelightDataset(path=Path('..') / 'RadianceFieldStudio' / 'data' / 'tensoir' / 'lego'),
            cuda=0,
        ),
        tsir_ficus=Tester(
            material=Path('out') / 'tsir_ficus' / 'validate',
            dataset=RelightDataset(path=Path('..') / 'RadianceFieldStudio' / 'data' / 'tensoir' / 'ficus'),
            cuda=0,
        ),
        tsir_hotdog=Tester(
            material=Path('out') / 'tsir_hotdog' / 'validate',
            dataset=RelightDataset(path=Path('..') / 'RadianceFieldStudio' / 'data' / 'tensoir' / 'hotdog'),
            cuda=0,
        ),
        tsir_arm=Tester(
            material=Path('out') / 'tsir_arm' / 'validate',
            dataset=RelightDataset(path=Path('..') / 'RadianceFieldStudio' / 'data' / 'tensoir' / 'armadillo'),
            cuda=0,
        ),
    ).run()
