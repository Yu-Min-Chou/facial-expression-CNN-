import dlib
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import numpy as np
import glob
import cv2 as cv


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.emotion_classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(num_classes, 7),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.emotion_classifier(x)

        return F.log_softmax(x, dim=1)


def main():

    label = ['Surprise', 'Fear', 'Disgust','Happy', 'Sadness', 'Anger', 'Neutral']

    parser = argparse.ArgumentParser(description='Facial expression by Corn')
    parser.add_argument('--shape_predictor', type=str, default='/home/guest/anaconda3/lib/python3.6/site-packages/face_recognition_models/models/shape_predictor_5_face_landmarks.dat', metavar='N',
                        help='the path of dlib.shape_predictor_5_face_landmarks.dat')
    parser.add_argument('--imagefolder_path', type=str, default='test', metavar='N',
                        help='the image folder for testing (default: test)(the picture should be *.jpg or *.png)')
    parser.add_argument('--dst_path', type=str, default='done', metavar='N',
                        help='the folder for the image which has been recognized (default: done)(please create a folder named [done] before you run this code)')
    args = parser.parse_args()

    image_folder = args.imagefolder_path
    dst_folder = args.dst_path

    #load detector to find a face
    #load sp to find face landmark and localize the face
    detector = dlib.get_frontal_face_detector()
    sp = dlib.shape_predictor('/home/guest/anaconda3/lib/python3.6/site-packages/face_recognition_models/models/shape_predictor_5_face_landmarks.dat')

    model = torch.load('trained_model/trained_model1')
    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # load picutres
    images = glob.iglob(image_folder+'/*jpg')

    for image in images:

        img = dlib.load_rgb_image(image)

        #detector find the bounding boxes of each face
        dets = detector(img,1)

        #check the faces in the image
        num_faces = len(dets)
        if(num_faces == 0):
            print('Sorry,we can not find any face')
            continue

        faces = dlib.full_object_detections()
        for detection in dets:
            faces.clear()
            faces.append(sp(img,detection))

            # crop the faces
            inputs = dlib.get_face_chips(img, faces, size=320)

            # facial expression analysis
            for input in inputs:
                im = Image.fromarray(input)
                im = data_transform(im)
                im = torch.unsqueeze(im, dim=0).cuda()
                output = model(im).cpu().detach().numpy()[0]
                output = np.exp(output)
                result = np.argmax(output)

                x = detection.left()
                y = detection.top()
                w = detection.right() - detection.left()
                h = detection.bottom() - detection.top()

                # connect result and faces
                #cv.line(img, (int(x + w / 2), y + 10), (x + w, y - 20), (255, 255, 255), 1)
                #cv.line(img, (x + w, y - 20), (x + w + 10, y - 20), (255, 255, 255), 1)

                #draw a transparent rectangle
                overlay = img.copy()
                opacity = 0.35
                cv.rectangle(img,(x - w + 10, y + 35),(x - w + 150,y + 175),(64,64,64),cv.FILLED)
                cv.addWeighted(img, 1 - opacity, overlay, opacity, 0, img)

                # print the result of the image
                print(image)
                for i in range(len(output)):
                    print('Emotion : {} , Possibility : {:.2f}%'.format(label[i], output[i] * 100))
                print('Result: {}'.format(label[result]))
                print('------------------------------------------')

                # draw a chart behind faces
                for i in range(len(output)):
                    if i == result:
                        color = (255,20,0)
                    else:
                        color = (0,255,255)

                    #length = int(110 * (round(output[i])))
                    number = round(output[i]*100,2)
                    adjustment = 3
                    rectangle_height = 15

                    #rectangle_1 = ((x + w + 40), (y - 25 + adjustment + i * 20))
                    #rectangle_2 = ((x + w + 40 + length), (y - 25 + adjustment + i * 20 + rectangle_height))
                    #cv.rectangle(img, rectangle_1, rectangle_2, color, cv.FILLED)

                    text = label[i] + ' : ' + str(number) + '%'
                    cv.putText(img, text, ((x - w + 10), (y + 40 + adjustment + i * 20)), cv.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        # RGB to BGR
        img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
        cv.imwrite(dst_folder + image[4:], img)


if __name__ == '__main__':
    main()