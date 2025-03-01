from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np
import nvdiffrast.torch as dr
import torch
from jaxtyping import Float32
from torch import Tensor

from rfstudio.data import RelightDataset
from rfstudio.engine.task import Task
from rfstudio.graphics import Cameras, PBRImages, TextureCubeMap, TextureLatLng, RGBImages
from rfstudio.io import dump_float32_image, load_float32_image, load_float32_masked_image


def refine_envmap(envmap: Tensor) -> Tensor:
    np_img = (envmap.clamp(0, 1) * 255).cpu().numpy().astype(np.uint8)
    np_img = cv2.resize(np_img, (800, 400), interpolation=cv2.INTER_LINEAR)
    return torch.nn.functional.pad(torch.from_numpy(np_img).to(envmap) / 255, (0, 0, 0, 0, 200, 200), value=1)

@dataclass
class Ours:

    scene: str

    @property
    def base_path(self) -> Path:
        return Path(f'out/s4r_{self.scene}/validate')

    def render(self, idx: int) -> Tensor:
        return RGBImages([load_float32_image(self.base_path / f'val_{idx:06d}_opt.png')]).resize_to(800, 800).item()

    def albedo(self, idx: int) -> Tensor:
        return load_float32_image(self.base_path.parent / 'envmap6' / f'val_{idx:06d}_kd.png')

    def roughness(self, idx: int) -> Tensor:
        return load_float32_image(self.base_path.parent / 'envmap6' / f'val_{idx:06d}_ks.png')[..., 1:2].repeat(1, 1, 3)

    def envmap(self) -> Tensor:
        latlng = TextureLatLng(data=load_float32_image(self.base_path / 'val_000000_light_image.png')).cuda()
        cubemap = latlng.as_cubemap(resolution=512)
        cubemap.z_up_to_y_up_()
        cubemap.rotateY_(-torch.pi / 2)
        latlng = cubemap.as_latlng(apply_transform=True, width=800, height=400)
        return latlng.data

@dataclass
class Ref:

    scene: str

    @property
    def base_path(self) -> Path:
        return Path(f'../RadianceFieldStudio/data/Synthetic4Relight/{self.scene}')

    def render(self, idx: int) -> Tensor:
        return load_float32_masked_image(self.base_path / 'test' / f'{idx:03d}_rgba.png')[..., :3]

    def albedo(self, idx: int) -> Tensor:
        return load_float32_masked_image(self.base_path / 'test' / f'{idx:03d}_albedo.png')[..., :3]

    def roughness(self, idx: int) -> Tensor:
        return load_float32_masked_image(self.base_path / 'test' / f'{idx:03d}_rough.png')[..., :3]

    def envmap(self) -> Tensor:
        return load_float32_image(Path('data/Synthetic4Relight/envmap3.exr'))

    def relight1(self, idx: int) -> Tensor:
        return load_float32_masked_image(self.base_path / 'test_rli' / f'envmap6_{idx:03d}.png')[..., :3]

    def relight2(self, idx: int) -> Tensor:
        return load_float32_masked_image(self.base_path / 'test_rli' / f'envmap12_{idx:03d}.png')[..., :3]

    def mask(self, idx: int) -> Tensor:
        return load_float32_masked_image(self.base_path / 'test' / f'{idx:03d}_rgba.png')[..., 3:]



@dataclass
class Qual(Task):

    output: Path = Path('exports') / 'relight'

    def _render_cubemap(
        self,
        cubemap: Float32[Tensor, "6 R R 3"],
        camera: Cameras,
        transform: Optional[Float32[Tensor, "3 3"]],
    ) -> Tensor:
        assert camera.shape == ()
        if transform is None:
            transform = torch.eye(3).to(camera.c2w)
        pixel_coords = camera.pixel_coordinates
        offset_y = (0.5 - camera.cy + pixel_coords[..., 0]) / camera.fy
        offset_x = (0.5 - camera.cx + pixel_coords[..., 1]) / camera.fx
        directions = (transform @ camera.c2w[:3, :3] @ torch.stack((
            offset_x,
            -offset_y,
            -torch.ones_like(offset_x),
        ), dim=-1)[..., None]).squeeze(-1) # [H, W, 3]
        directions = directions / directions.norm(dim=-1, keepdim=True)
        img = dr.texture(cubemap[None] ** (1 / 2.2), directions[None].contiguous(), filter_mode='linear', boundary_mode='cube')[0]
        return img

    def run(self) -> None:
        for view, scene in zip(
            # [0, 37, 10, 43],
            # ['air_baloons', 'chair', 'hotdog', 'jugs'],
            [7, 43],
            ['hotdog', 'jugs'],
        ):
            base_dir = self.output / f's4r_{scene}' if scene != 'air_baloons' else self.output / 's4r_air'
            base_dir.mkdir(exist_ok=True, parents=True)
            # dataset = RelightDataset(Path('data/Synthetic4Relight') / scene)
            # dataset.__setup__()
            # dataset.to(self.device)
            # camera = dataset.get_inputs(split='test')[view]
            baselines: Dict[str, Ours] = {
                # 'r3dg': R3DG(scene),
                # 'nv': NVdiffrec(scene),
                'nv': Ours(scene),
                # 'gsir': GSIR(scene),
            }
            ref = Ref(scene)
            mask = ref.mask(view).cpu()
            bg = 1
            # light_bg1 = TextureCubeMap.from_image_file(
            #     Path('data/Synthetic4Relight/envmap6.exr'),
            #     resolution=1024,
            #     device=self.device,
            # ).render(camera).item().cpu().clamp(0, 1)
            # light_bg2 = TextureCubeMap.from_image_file(
            #     Path('data/Synthetic4Relight/envmap12.exr'),
            #     resolution=1024,
            #     device=self.device,
            # ).render(camera).item().cpu().clamp(0, 1)
            for name, method in baselines.items():
                dump_float32_image(base_dir / f'{name}_albedo.png', method.albedo(view) * mask + bg * (1 - mask))
                dump_float32_image(base_dir / f'{name}_rgb.png', method.render(view) * mask + bg * (1 - mask))
                dump_float32_image(base_dir / f'{name}_roughness.png', method.roughness(view) * mask + bg * (1 - mask))
                dump_float32_image(base_dir / f'{name}_env.png', refine_envmap(method.envmap()))
                # if name == 'Ours':
                #     dump_float32_image(base_dir / f'{name}_reli_env1.png', method.relight1(view))
                #     dump_float32_image(base_dir / f'{name}_reli_env2.png', method.relight2(view))
                # else:
                #     dump_float32_image(base_dir / f'{name}_reli_env1.png', method.relight1(view) * mask + light_bg1 * (1 - mask))
                #     dump_float32_image(base_dir / f'{name}_reli_env2.png', method.relight2(view) * mask + light_bg2 * (1 - mask))
            # dump_float32_image(base_dir / 'gt_albedo.png', ref.albedo(view) * mask + bg * (1 - mask))
            # dump_float32_image(base_dir / 'gt_rgba.png', ref.render(view) * mask + bg * (1 - mask))
            # dump_float32_image(base_dir / 'gt_roughness.png', ref.roughness(view) * mask + bg * (1 - mask))
            # dump_float32_image(base_dir / 'gt_env.png', refine_envmap(ref.envmap()))
            # dump_float32_image(base_dir / 'gt_reli_env1.png', ref.relight1(view) * mask + light_bg1 * (1 - mask))
            # dump_float32_image(base_dir / 'gt_reli_env2.png', ref.relight2(view) * mask + light_bg2 * (1 - mask))

if __name__ == '__main__':
    Qual(cuda=0).run()
