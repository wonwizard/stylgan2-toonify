#image resize
#made by won.wizard

from PIL import Image
from glob import glob
import os
import argparse


parser = argparse.ArgumentParser(description='image resize')

# 입력받을 인자값 등록
parser.add_argument('--src_dir', required=True,  help='source image directory. example ./src_image')
parser.add_argument('--target_dir', required=True, help='target image directory. example ./target_image')
parser.add_argument('--size', required=False, default='512', help='resize size. default 512')

# 입력받은 인자값을 args에 저장 (type: namespace)
args = parser.parse_args()

src_dir = args.src_dir
target_dir= args.target_dir
size = int(args.size)

#files = glob(src_dir+'/*')

files_PNG = glob(src_dir+'/*.PNG')
files_JPG = glob(src_dir+'/*.JPG')
files_png = glob(src_dir+'/*.png')
files_jpg = glob(src_dir+'/*.jpg')
files_JPEG = glob(src_dir+'/*.JPEG')
files_jpeg = glob(src_dir+'/*.jpeg')

files = files_PNG+files_JPG+files_png+files_jpg+files_JPEG+files_jpeg

print('start resize.',size,'size',len(files),'files. from', src_dir,'to',target_dir) 

num = 0
for f in files : 
   img = Image.open(f)
   img_resize_lanczos = img.resize((size, size), Image.LANCZOS)
   dir_name, file_name = os.path.split(f)
   img_resize_lanczos.save(target_dir+os.path.sep+file_name)
   num += 1

print(num,"complete.")

