#https://github.com/a312863063/generators-with-stylegan2
#lnw changed

import pickle
import PIL.Image
import numpy as np
import dnnlib
import dnnlib.tflib as tflib
import os

import argparse
import glob
from natsort import natsorted
from moviepy.editor import *

from PIL import ImageFont
from PIL import ImageDraw



def move_latent_and_save(latent_vector, direction_file, coeffs, Gs_network, Gs_syn_kwargs):
    direction = np.load('latent_directions/' + direction_file)
    os.makedirs('results/'+direction_file.split('.')[0], exist_ok=True)
    '''latent_vector는 얼굴의 잠재 인코딩, 방향은 얼굴 조정 방향, coeffs는변경 정도'''
    for i, coeff in enumerate(coeffs):

        #lwn for debug
        #print(latent_vector.shape)   #-> (1, 18, 512)
        #print(direction.shape)      # -> (18, 512)

        new_latent_vector = latent_vector.copy()
        #new_latent_vector[0][:8] = (latent_vector[0] + coeff*direction)[:8]
        new_latent_vector[:8] = (latent_vector + coeff*direction)[:8]

        images = Gs_network.components.synthesis.run(new_latent_vector, **Gs_syn_kwargs)
        result = PIL.Image.fromarray(images[0], 'RGB')

        #lnw add
        font = ImageFont.truetype("times.ttf", 40)
        #font = ImageFont.truetype("times.ttf", 35)
        #font = ImageFont.truetype(os.path.join(fontsFolder,'Zapfino.ttf'),15)
        draw = ImageDraw.Draw(result)
        draw.text((400,950),direction_file.split('.')[0],(0,0,0),font=font)
        #draw.text((30,30),direction_file.split('.')[0],(0,0,0),font=font)

        result.save('results/'+direction_file.split('.')[0]+'/'+str(i).zfill(3)+'.png')


#----------------------------------------------------------------------------

_examples = '''examples:

  # Generate latents direction image and mov
  python %(prog)s --dlatents=project_gen/0001-proj.npz --outdir=./results --network=../stylegan2_face_seepretty_lnwedit/generator_star-stylegan2-config-f.pkl

'''

#----------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description='Generate images using pretrained network pickle.',
        epilog=_examples
    )

    parser.add_argument('--network', help='Network pickle filename', required=True)
    parser.add_argument('--dlatents', help='Generate images for saved dlatents')
    parser.add_argument('--trunc', type=float, help='Truncation psi (default: %(default)s)', default=0.5)
    parser.add_argument('--outdir', help='Where to save the output images', required=True, metavar='DIR')

    args = parser.parse_args()

    #lnw add
    from datetime import datetime
    start_time = datetime.now()

    tflib.init_tf()
    #with open('../stylegan2_face_seepretty_lnwedit/generator_star-stylegan2-config-f.pkl', "rb") as f:
    with open(args.network, "rb") as f:
        generator_network, discriminator_network, Gs_network = pickle.load(f)

    w_avg = Gs_network.get_var('dlatent_avg')
    noise_vars = [var for name, var in Gs_network.components.synthesis.vars.items() if name.startswith('noise')]
    Gs_syn_kwargs = dnnlib.EasyDict()
    Gs_syn_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    Gs_syn_kwargs.randomize_noise = False
    Gs_syn_kwargs.minibatch_size = 1
    #truncation_psi = 0.5
    truncation_psi = args.trunc

    #w = np.load('project_gen/0001-proj.npz')['dlatents']
    # if npz
    #w = np.load(args.dlatents)['dlatents']
    # if npy
    w = np.load(args.dlatents)
    w = np.expand_dims(w,axis=0)

    direction_file_eye = 'eyes_open.npy' 
    direction_file_hor = 'angle_horizontal.npy'
    direction_file_pit = 'angle_pitch.npy'
    direction_file_sad = 'emotion_sad.npy'
    direction_file_sur = 'emotion_surprise.npy'
    direction_file_ang = 'emotion_angry.npy'
    direction_file_happy = 'emotion_happy.npy'

    #coeffs = [-15., -12., -9., -6., -3., 0., 3., 6., 9., 12.]
    coeffs1 = [0., 0., 2., 4., 6., 8., 10., 8., 6., 4., 2., 0., 0., 0., 0.]   # for eyes
    coeffs2 = [0., 0., 0., 0., 3., 6., 9., 6., 3., 0., 0., 0., 0., 0., 0.]   # for fast eyes
    coeffs22 = [0., 0., 0., 0., 0., 5., 11., 5., 0., 0., 0., 0., 0., 0., 0.]   # for fast eyes2
    coeffs3 = [0., 0., 0., -1., -2., -3., -4., -5., -4., -3., -2., -1., 0., 0., 0.]  # for angle_horizontal  pich / head down
    coeffs4 = [0., 0., 0., -2., -4., -6., -7., -7., -7., -3.5, 0., 0., 0., 0., 0.] # for many pich / head down
    coeffs5 = [0., 0., 0., 1., 2., 3., 4., 5., 5, 5., 2.5, 0., 0., 0., 0.]   # for happy sad surprise 
    coeffs6 = [0., 0., 0., 2., 4., 6., 8., 10., 8., 6., 4., 2., 0., 0., 0.]   # for angly  many angle_horizontal
    coeffs62 = [0., 0., 0., 2., 4., 6., 8., 8., 8., 4., 0., 0., 0., 0., 0.]   # for angly small angle_horizontal


    #1) happy
    direction_file = direction_file_happy
    coeffs = coeffs5
    move_latent_and_save(w, direction_file, coeffs, Gs_network, Gs_syn_kwargs)

    #lnw add
    end_time = datetime.now()
    process_time = end_time - start_time
    print('process_time:',process_time )


    #2) eyes
    direction_file = direction_file_eye
    coeffs = coeffs22
    move_latent_and_save(w, direction_file, coeffs, Gs_network, Gs_syn_kwargs)

    #lnw add
    end_time = datetime.now()
    process_time = end_time - start_time
    print('process_time:',process_time )


    #3) horizontal
    direction_file = direction_file_hor
    coeffs = coeffs62
    move_latent_and_save(w, direction_file, coeffs, Gs_network, Gs_syn_kwargs)

    #lnw add
    end_time = datetime.now()
    process_time = end_time - start_time
    print('process_time:',process_time )

    #4) head down
    direction_file = direction_file_pit
    coeffs = coeffs4
    move_latent_and_save(w, direction_file, coeffs, Gs_network, Gs_syn_kwargs)

    #lnw add
    end_time = datetime.now()
    process_time = end_time - start_time
    print('process_time:',process_time )


    #5) angly
    direction_file = direction_file_ang
    coeffs = coeffs62
    move_latent_and_save(w, direction_file, coeffs, Gs_network, Gs_syn_kwargs)

    #lnw add
    end_time = datetime.now()
    process_time = end_time - start_time
    print('process_time:',process_time )


    
    # make imgs to mp4 

    gif_name = 'pic'
    fps = 15

    total_file_list = []

    dir_name = args.outdir+"/emotion_happy/"
    file_list = glob.glob(dir_name+'*.png')  # Get all the pngs in the current directory
    file_list_sorted = natsorted(file_list,reverse=False)  # Sort the images
    total_file_list.extend(file_list)

    dir_name = args.outdir+"/emotion_angry/"
    file_list = glob.glob(dir_name+'*.png')  # Get all the pngs in the current directory
    file_list_sorted = natsorted(file_list,reverse=False)  # Sort the images
    total_file_list.extend(file_list)

    dir_name = args.outdir+"/eyes_open/"
    file_list = glob.glob(dir_name+'*.png')  # Get all the pngs in the current directory
    file_list_sorted = natsorted(file_list,reverse=False)  # Sort the images
    total_file_list.extend(file_list)

    dir_name = args.outdir+"/angle_pitch/"
    file_list = glob.glob(dir_name+'*.png')  # Get all the pngs in the current directory
    file_list_sorted = natsorted(file_list,reverse=False)  # Sort the images
    total_file_list.extend(file_list)

    dir_name = args.outdir+"/angle_horizontal/"
    file_list = glob.glob(dir_name+'*.png')  # Get all the pngs in the current directory
    file_list_sorted = natsorted(file_list,reverse=False)  # Sort the images
    total_file_list.extend(file_list)

    clips = [ImageClip(m).set_duration(0.1)
            #for m in file_list_sorted]
            for m in total_file_list]

    concat_clip = concatenate_videoclips(clips, method="compose")

    concat_clip.write_videofile(args.outdir+"/"+os.path.basename(args.dlatents)[:-4]+"-movie.mp4", fps=fps)


#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------

