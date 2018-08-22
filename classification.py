import shutil
import os
from PIL import Image

label = open('emotion/basic face/EmoLabel/list_patition_label.txt','r')
for line in label:
    picture_name = ""
    first_space = False
    number = -1
    for word in line:
        if(first_space):
            number = word
            break
        else:
            if(word.isspace()):
                first_space = True
            else:
                picture_name = picture_name + word

    length = len(picture_name)
    im = Image.open('emotion/basic face/Image/aligned/'+picture_name[:length-4]+'_aligned.jpg')
    dst = "emotion/basic face/Image/00"+number
    jpgfile = os.path.join('emotion/basic face/Image/aligned',picture_name[:length-4]+'_aligned.jpg')
    shutil.copy(jpgfile,dst)
    print("success")