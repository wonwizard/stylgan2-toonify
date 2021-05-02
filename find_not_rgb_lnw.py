from glob import glob
from PIL import Image,ImageChops

files = glob('./train_data_photo/*')

no_file = 0

for f in files:
    im = Image.open(f)
    if im.mode not in ("L", "RGB"):
        print(f)
        no_file += 1

print('complete. No RGB File ',no_file)
