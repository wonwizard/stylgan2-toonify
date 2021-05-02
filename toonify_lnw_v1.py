#https://github.com/justinpinkney/toonify/blob/master/toonify-yourself.ipynb
import pretrained_networks

#blended_url = "ffhq-cartoon-blended-64.pkl"
#blended_url = "blended_t4_snapshpt577_anime_star_64.pkl"  
#blended_url = "blended_t3_snapshpt24_anime_star_128.pkl"
#blended_url = "blended_t3_snapshpt24_anime_star_64.pkl"
#blended_url = "blended_snapshpt48_anime_star_64.pkl"
#blended_url = "blended_snapshpt48_anime_star_128.pkl"
#blended_url = "blended_snapshpt96_anime_star_64.pkl"
#blended_url = "blended_snapshpt96_anime_star_128.pkl"
#blended_url = "blended_snapshpt24_star_anime_64.pkl"
#blended_url = "blended_snapshpt24_star_anime_128.pkl" 
#blended_url = "blended_t4_snapshpt24_star_anime_32.pkl"
#blended_url = "blended_snapshpt48_star_anime_64.pkl"     # file none
#blended_url = "blended_snapshpt48_star_anime_128.pkl"
#blended_url = "blended_snapshpt48_star_anime_32.pkl"    
#blended_url = "blended_snapshpt72_star_anime_64.pkl"
#blended_url = "blended_snapshpt72_star_anime_128.pkl"
#blended_url = "blended_snapshpt72_star_anime_32.pkl"   # file none
#blended_url = "blended_t4_snapshpt577_star_anime_64.pkl"
#blended_url = "blended_t4_snapshpt577_star_anime_128.pkl"
#blended_url = "blended_t4_snapshpt577_star_anime_32.pkl"

#0419
#blended_url = "blended_t6_snapshpt24_anime_star_128.pkl"
#blended_url = "blended_t6_snapshpt30_anime_star_64.pkl"
#blended_url = "blended_t6_snapshpt30_anime_star_128.pkl"
#blended_url = "blended_t6_snapshpt36_anime_star_64.pkl"
#blended_url = "blended_t6_snapshpt36_anime_star_128.pkl"
#blended_url = "blended_t6_snapshpt42_anime_star_64.pkl"
#blended_url = "blended_t6_snapshpt42_anime_star_128.pkl"
#blended_url = "blended_t6_snapshpt48_anime_star_64.pkl"
#blended_url = "blended_t6_snapshpt90_anime_star_64.pkl"


# "ffhq-cartoon-blended-64.pkl"  https://drive.google.com/uc?id=1H73TfV5gQ9ot7slSed_l-lim9X7pMRiU
#ffhq_url = "../stylegan2_face_seepretty_lnwedit/generator_star-stylegan2-config-f.pkl"   
# http://d36zk2xti64re0.cloudfront.net/stylegan2/networks/stylegan2-ffhq-config-f.pkl


import argparse

import numpy as np
from PIL import Image
import dnnlib
import dnnlib.tflib as tflib
from pathlib import Path

# 인자값을 받을 수 있는 인스턴스 생성
parser = argparse.ArgumentParser(description='toonify')

# 입력받을 인자값 등록
parser.add_argument('--blendednet', required=True, help='input blended network model')
#blendednet = "blended_t6_snapshpt90_anime_star_64.pkl"
parser.add_argument('--facenet', required=False, default='../stylegan2_face_seepretty_lnwedit/generator_star-stylegan2-config-f.pkl', help='input face network model')
#facenet = "../stylegan2_face_seepretty_lnwedit/generator_star-stylegan2-config-f.pkl"
# "ffhq-cartoon-blended-64.pkl"  https://drive.google.com/uc?id=1H73TfV5gQ9ot7slSed_l-lim9X7pMRiU
# http://d36zk2xti64re0.cloudfront.net/stylegan2/networks/stylegan2-ffhq-config-f.pkl
parser.add_argument('--image_dir', required=False, default='project_gen', help='image load and save dir, default project_gen/')
parser.add_argument('--filename', required=False, default='-toon.jpg', help='save file name. default -toon.jpg')

# 입력받은 인자값을 args에 저장 (type: namespace)
args = parser.parse_args()

# 입력받은 인자값 출력
#print(args.blendednet)
#print(args.facenet)
#print(args.image_dir)
#print(args.filename)

blended_net = args.blendednet
face_net = args.facenet
imagedir = args.image_dir
file_name = args.filename

print("blended_net",blended_net)
print("face_net",face_net)
print("imagedir",imagedir)
print("file_name",file_name)

_, _, Gs_blended = pretrained_networks.load_networks(blended_net)
_, _, Gs = pretrained_networks.load_networks(face_net)


#latent_dir = Path("project_gen")
latent_dir = Path(imagedir)

latents = latent_dir.glob("*.npy")

for latent_file in latents:
    latent = np.load(latent_file)
    latent = np.expand_dims(latent,axis=0)
    synthesis_kwargs = dict(output_transform=dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=False), minibatch_size=8)
    images = Gs_blended.components.synthesis.run(latent, randomize_noise=False, **synthesis_kwargs)
    Image.fromarray(images.transpose((0,2,3,1))[0], 'RGB').save(latent_file.parent / (f"{latent_file.stem}"+file_name))



