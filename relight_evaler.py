from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Type

from rfstudio.engine.task import Task, TaskGroup
from rfstudio.graphics import Cameras, Texture2D, TextureCubeMap, TriangleMesh, RGBAImages, RGBImages
from rfstudio.graphics.shaders import BaseShader, ShadingContext, NormalShader
from rfstudio.io import dump_float32_image, load_float32_image, load_float32_masked_image
from rfstudio.data import RelightDataset
from rfstudio.visualization import TabularFigures
from rfstudio.ui import console
from rfstudio.loss import PSNRLoss, SSIMLoss, LPIPSLoss
from rfstudio.utils.pretty import P

from jaxtyping import Float32
from torch import Tensor
import torch
import numpy as np


@dataclass
class RelightEvaler(Task):

    name: str = ...

    dataset: RelightDataset = RelightDataset(path=...)

    skip_rlit: bool = False

    skip_mat: bool = False

    @torch.no_grad()
    def run(self) -> None:
        self.dataset.to(self.device)
        (
            gt_albedos,
            gt_roughnesses,
            gt_relights,
            gt_relight_envmaps,
        ) = self.dataset.get_meta(split='test')
        if not self.skip_rlit:
            for relight_idx, relights in enumerate(gt_relights):
                light_name = gt_relight_envmaps[relight_idx].stem
                relight_idx += 1
                gt_rgb = relights[...].blend((1, 1, 1)).clamp(0, 1)
                rgb = RGBImages(torch.stack([
                    load_float32_image(Path('out') / self.name / light_name / f'val_{i:06d}_opt.png')
                    for i in range(200)
                ])).to(self.device)
                psnr = PSNRLoss()(rgb, gt_rgb)
                ssim = 1 - SSIMLoss()(rgb, gt_rgb)
                lpips = LPIPSLoss()(rgb, gt_rgb)
                console.print(P@'RLIT[{relight_idx}] @ PSNR: {psnr:.3f}')
                console.print(P@'RLIT[{relight_idx}] @ SSIM: {ssim:.4f}')
                console.print(P@'RLIT[{relight_idx}] @ LPIPS: {lpips:.4f}')
        if not self.skip_mat:
            if gt_roughnesses is not None:
                roughness = torch.stack([
                    load_float32_image(
                        Path('out') / self.name / light_name / f'val_{i:06d}_ks.png'
                    )[..., 1:2].to(self.device) * gt_albedo.item()[..., 3:]
                    for i, gt_albedo in enumerate(gt_albedos)
                ]) # [H, W, 1]
                gt_roughnesses = torch.stack([item[..., 0:1] for item in gt_roughnesses[...].blend((0, 0, 0))])
                roughness_mse = (
                    torch.nn.functional.mse_loss(
                        roughness,
                        gt_roughnesses,
                    )
                )
            albedos = RGBImages([
                load_float32_image(
                    Path('out') / self.name / light_name / f'val_{i:06d}_kd.png'
                ).to(self.device) * gt_albedo.item()[..., 3:]
                for i, gt_albedo in enumerate(gt_albedos)
            ])
            gt_albedos = gt_albedos[...].blend((0, 0, 0))
            psnr = PSNRLoss()(albedos, gt_albedos)
            ssim = 1 - SSIMLoss()(albedos, gt_albedos)
            lpips = LPIPSLoss()(albedos, gt_albedos)
            console.print(P@'Albedo @ PSNR: {psnr:.3f}')
            console.print(P@'Albedo @ SSIM: {ssim:.4f}')
            console.print(P@'Albedo @ LPIPS: {lpips:.4f}')
            if gt_roughnesses is not None:
                console.print(P@'Roughness @ MSE: {roughness_mse:.3f}')
            else:
                console.print(P@'Roughness @ MSE: N/A')


if __name__ == '__main__':
    TaskGroup(
        s4r_air=RelightEvaler(
            dataset=RelightDataset(path=Path('..') / 'RadianceFieldStudio' / 'data' / 'Synthetic4Relight' / 'air_baloons'),
            name='s4r_air',
            cuda=0,
        ),
        s4r_chair=RelightEvaler(
            dataset=RelightDataset(path=Path('..') / 'RadianceFieldStudio' / 'data' / 'Synthetic4Relight' / 'chair'),
            name='s4r_chair',
            cuda=0,
        ),
        s4r_hotdog=RelightEvaler(
            dataset=RelightDataset(path=Path('..') / 'RadianceFieldStudio' / 'data' / 'Synthetic4Relight' / 'hotdog'),
            name='s4r_hotdog',
            cuda=0,
        ),
        s4r_jugs=RelightEvaler(
            dataset=RelightDataset(path=Path('..') / 'RadianceFieldStudio' / 'data' / 'Synthetic4Relight' / 'jugs'),
            name='s4r_jugs',
            cuda=0,
        ),
        tsir_hotdog=RelightEvaler(
            dataset=RelightDataset(path=Path('..') / 'RadianceFieldStudio' / 'data' / 'tensoir' / 'hotdog'),
            name='tsir_hotdog',
            cuda=0,
        ),
        tsir_ficus=RelightEvaler(
            dataset=RelightDataset(path=Path('..') / 'RadianceFieldStudio' / 'data' / 'tensoir' / 'ficus'),
            name='tsir_ficus',
            cuda=0,
        ),
        tsir_arm=RelightEvaler(
            dataset=RelightDataset(path=Path('..') / 'RadianceFieldStudio' / 'data' / 'tensoir' / 'armadillo'),
            name='tsir_arm',
            cuda=0,
        ),
        tsir_lego=RelightEvaler(
            dataset=RelightDataset(path=Path('..') / 'RadianceFieldStudio' / 'data' / 'tensoir' / 'lego'),
            name='tsir_lego',
            cuda=0,
        ),
    ).run()