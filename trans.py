from PIL import Image
from os import listdir
import glob


for name in glob.glob('*.tiff'):
    im = Image.open(name).convert('RGB')
    name = str(name).rstrip(".tiff")
    im.save(name + '.jpg', 'JPEG')

print ("Conversion from tif/tiff to jpg completed!")