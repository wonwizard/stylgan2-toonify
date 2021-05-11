# add text in image 

from PIL import Image
from glob import glob
import os
from PIL import ImageFont
from PIL import ImageDraw
import argparse

parser = argparse.ArgumentParser(description='add text in image')

# 입력받을 인자값 등록
parser.add_argument('--src_dir', required=True,  help='source image directory. example ./src_image')
parser.add_argument('--target_dir', required=True, help='target image directory. example ./target_image')
parser.add_argument('--text', required=True, help='text')

# 입력받은 인자값을 args에 저장 (type: namespace)
args = parser.parse_args()

#font = ImageFont.truetype("timesbd.ttf", 40)
#font = ImageFont.truetype("timesbd.ttf", 60)
#font = ImageFont.truetype("malgunbd.ttf", 70) #맑은 고딕 굵음
font = ImageFont.truetype("HMKMRHD.TTF", 80) #휴먼둥근헤드라인 


src_dir = args.src_dir
target_dir= args.target_dir
text = args.text

#files = glob(src_dir+'/*')

files_PNG = glob(src_dir+'/*.PNG')
files_JPG = glob(src_dir+'/*.JPG')
files_png = glob(src_dir+'/*.png')
files_jpg = glob(src_dir+'/*.jpg')
files_JPEG = glob(src_dir+'/*.JPEG')
files_jpeg = glob(src_dir+'/*.jpeg')

files = files_PNG+files_JPG+files_png+files_jpg+files_JPEG+files_jpeg

print('start add text.',text,' files. from', src_dir,'to',target_dir)

num = 0
for f in files :
   img = Image.open(f)
   draw = ImageDraw.Draw(img)
   #draw.text((100,400),text,(128,128,128),font=font)
   #draw.text((400,800),text,(128,128,128),font=font)
   draw.text((330,200),text,(128,128,128),font=font)
   dir_name, file_name = os.path.split(f)
   #img.save(target_dir+os.path.sep+'mark'+file_name)
   img.save(target_dir+os.path.sep+'mark'+file_name, quality=100, subsampling=0)
   num += 1

print(num,"complete.")
