#https://github.com/justinpinkney/toonify/blob/master/toonify-yourself.ipynb

import pretrained_networks
import argparse

import numpy as np
from PIL import Image
import dnnlib
import dnnlib.tflib as tflib
from pathlib import Path

parser = argparse.ArgumentParser(description='toonify')

parser.add_argument('--blendednet', required=True, help='input blended network model')
parser.add_argument('--image_dir', required=False, default='data_project_gen', help='image load and save dir (default: data_project_gen)')
parser.add_argument('--filename', required=False, default='_toon.jpg', help='save file name. (default: _toon.jpg)')


args = parser.parse_args()

blended_net = args.blendednet
imagedir = args.image_dir
file_name = args.filename

print("blended_net",blended_net)
print("imagedir",imagedir)
print("file_name",file_name)

_, _, Gs_blended = pretrained_networks.load_networks(blended_net)
#_, _, Gs = pretrained_networks.load_networks(face_net)


latent_dir = Path(imagedir)

latents = latent_dir.glob("*.npy")

for latent_file in latents:
    latent = np.load(latent_file)
    latent = np.expand_dims(latent,axis=0)
    synthesis_kwargs = dict(output_transform=dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=False), minibatch_size=8)
    images = Gs_blended.components.synthesis.run(latent, randomize_noise=False, **synthesis_kwargs)
    Image.fromarray(images.transpose((0,2,3,1))[0], 'RGB').save(latent_file.parent / (f"{latent_file.stem}"+file_name))



