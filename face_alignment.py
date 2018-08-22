import dlib
import glob
from PIL import Image


#load detector to find a face
#load sp to find face landmark and localize the face
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor('/home/guest/anaconda3/lib/python3.6/site-packages/face_recognition_models/models/shape_predictor_5_face_landmarks.dat')


for number in range(1,8):
    num = number
    dst = "train_data2_alignment/00" + str(num)+'/'
    all_file = glob.iglob("train_data2/00"+str(num)+"/*.jpg")
    for picture in all_file:
        # load a image
        img = dlib.load_rgb_image(picture)
        # detector find the bounding boxes of each face
        dets = detector(img, 1)
        num_faces = len(dets)

        if (num_faces == 0):
            print('Sorry,we can not find any face in' + picture)
            continue

        faces = dlib.full_object_detections()

        for detection in dets:
            faces.append(sp(img, detection))

        images = dlib.get_face_chips(img, faces, size=320)

        for image in images:
            im = Image.fromarray(image)
            im.save(dst + picture[16:], 'JPEG')

        image = dlib.get_face_chip(img, faces[0])
        im = Image.fromarray(image)
        im.save(dst + picture[16:], 'JPEG')









