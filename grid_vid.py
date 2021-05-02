"""
Author: lzhbrian (https://lzhbrian.me)
Date: 2020.1.20
Note: mainly modified from: https://github.com/tkarras/progressive_growing_of_gans/blob/master/util_scripts.py#L50
"""

import numpy as np
from PIL import Image
import os
import scipy
import pickle
import moviepy
import dnnlib
import dnnlib.tflib as tflib
from tqdm import tqdm

from pathlib import Path
import typer



def load_net(fpath):
    tflib.init_tf()
    with open(fpath, 'rb') as stream:
        _G, _D, Gs = pickle.load(stream, encoding='latin1')

    return Gs
    
fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)

def create_image_grid(images, grid_size=None):
    assert images.ndim == 3 or images.ndim == 4
    num, img_w, img_h = images.shape[0], images.shape[-1], images.shape[-2]

    if grid_size is not None:
        grid_w, grid_h = tuple(grid_size)
    else:
        grid_w = max(int(np.ceil(np.sqrt(num))), 1)
        grid_h = max((num - 1) // grid_w + 1, 1)

    grid = np.zeros(list(images.shape[1:-2]) + [grid_h * img_h, grid_w * img_w], dtype=images.dtype)
    for idx in range(num):
        x = (idx % grid_w) * img_w
        y = (idx // grid_w) * img_h
        grid[..., y : y + img_h, x : x + img_w] = images[idx]
    return grid

    # grid_size=[4,4], mp4_fps=25, duration_sec=10.0, smoothing_sec=2.0, truncation_psi=0.7)
from typing import Tuple

def generate_interpolation_video(net: Path,
                                 mp4: Path = Path("output.mp4"), 
                                 truncation_psi:float =0.5,
                                 grid_size: Tuple[int, int]=(1,1), 
                                 duration_sec:float =60.0, 
                                 smoothing_sec:float =1.0, 
                                 mp4_fps:int=30, 
                                 mp4_codec='libx264',
                                 random_seed:int = 1000,
                                 minibatch_size:int = 8,
                                 output_width: int = typer.Option(None)):

    Gs = load_net(net)
    num_frames = int(np.rint(duration_sec * mp4_fps))
    random_state = np.random.RandomState(random_seed)

    print('Generating latent vectors...')
    shape = [num_frames, np.prod(grid_size)] + Gs.input_shape[1:] # [frame, image, channel, component]
    all_latents = random_state.randn(*shape).astype(np.float32)
    all_latents = scipy.ndimage.gaussian_filter(all_latents, [smoothing_sec * mp4_fps] + [0] * len(Gs.input_shape), mode='wrap')
    all_latents /= np.sqrt(np.mean(np.square(all_latents)))

    # Frame generation func for moviepy.
    def make_frame(t):
        frame_idx = int(np.clip(np.round(t * mp4_fps), 0, num_frames - 1))
        latents = all_latents[frame_idx]
        labels = np.zeros([latents.shape[0], 0], np.float32)        
        images = Gs.run(latents, None, truncation_psi=truncation_psi, randomize_noise=False, output_transform=fmt, minibatch_size=minibatch_size)
        
        images = images.transpose(0, 3, 1, 2) #NHWC -> NCHW
        grid = create_image_grid(images, grid_size).transpose(1, 2, 0) # HWC
        if grid.shape[2] == 1:
            grid = grid.repeat(3, 2) # grayscale => RGB
        return grid

    # Generate video.
    import moviepy.editor # pip install moviepy
    c = moviepy.editor.VideoClip(make_frame, duration=duration_sec)
    if output_width:
        c = c.resize(width=output_width)
    c.write_videofile(str(mp4), fps=mp4_fps, codec=mp4_codec)
    return c

if __name__ == "__main__":
    typer.run(generate_interpolation_video)