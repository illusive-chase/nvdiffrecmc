from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from rfstudio.engine.task import Task
from rfstudio.graphics import TextureCubeMap, TextureLatLng
from rfstudio.io import dump_float32_image
import cv2


@dataclass
class Tester(Task):

    envmap: Path = ...
    output: Path = ...

    def run(self) -> None:
        latlng = TextureLatLng.from_image_file(self.envmap, device=self.device)
        cubemap = latlng.as_cubemap(resolution=512)
        cubemap.rotateY_(torch.pi / 2)
        cubemap.y_up_to_z_up_()
        latlng = cubemap.as_latlng(apply_transform=True)
        cv2.imwrite(str(self.output), latlng.data.flip(-1).contiguous().cpu().numpy())

if __name__ == '__main__':
    Tester(cuda=0).run()