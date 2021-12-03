from __future__ import print_function

import argparse
import os
import sys

from PIL import Image
import cv2
import torch
from torch.autograd import Variable
import matplotlib.patches as patches
from data import VOCAnnotationTransform, VOCDetection, BaseTransform, VOC_CLASSES
from data import VOC_ROOT, VOC_CLASSES as labelmap
from ssd import build_ssd
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
from matplotlib import pyplot as plt
from PIL import ImageDraw

os.environ['KMP_DUPLICATE_LIB_OK']='True'

parser = argparse.ArgumentParser(description='Single Shot MultiBox Detection')
parser.add_argument('--trained_model', default='weights/ssd_finally.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--save_folder', default='eval/', type=str,
                    help='Dir to save results')
parser.add_argument('--visual_threshold', default=0.5, type=float,
                    help='Final confidence threshold')
parser.add_argument('--cuda', default=False, type=bool,
                    help='Use cuda to train model')
parser.add_argument('--voc_root', default=VOC_ROOT, help='Location of VOC root directory')
parser.add_argument('-f', default=None, type=str, help="Dummy arg so we can load in Jupyter Notebooks")
args = parser.parse_args()

if args.cuda and torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)


def test_net(save_folder, net, cuda, testset, transform, thresh):
    # dump predictions and assoc. ground truth to text file for now
    filename = save_folder+'test.txt'
    num_images = len(testset)
    for i in range(num_images):
        print('Testing image {:d}/{:d}....'.format(i+1, num_images))
        img = testset.pull_image(i)
        img_id, annotation = testset.pull_anno(i)
        
        imageFile='./data/VOC2007/JPEGImages/'+img_id+'.jpeg'
        image=cv2.imread(imageFile)
        rgb_image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        colors = plt.cm.hsv(np.linspace(0, 1, 3)).tolist()
        plt.imshow(rgb_image)
#         plt.show()
        currentAxis = plt.gca()
#         
        for k in range(len(annotation)):
            currentAxis.add_patch(patches.Rectangle((annotation[k][0],annotation[k][1]),annotation[k][2]-annotation[k][0],
                                                annotation[k][3]-annotation[k][1], linewidth=2,edgecolor='r',facecolor='none'))
        
        x = torch.from_numpy(transform(img)[0]).permute(2, 0, 1)
        x = Variable(x.unsqueeze(0))

        with open(filename, mode='a') as f:
            f.write('\nGROUND TRUTH FOR: '+img_id+'\n')
            for box in annotation:
                f.write('label: '+' || '.join(str(b) for b in box)+'\n')
        if cuda:
            x = x.cuda()

        y = net(x)      # forward pass
        detections = y.data
        # scale each detection back up to the image
        scale = torch.Tensor([img.shape[1], img.shape[0],
                             img.shape[1], img.shape[0]])
        pred_num = 0
        for i in range(detections.size(1)):
            j = 0
            while detections[0, i, j, 0] >= 0.4:
                if pred_num == 0:
                    with open(filename, mode='a') as f:
                        f.write('PREDICTIONS: '+'\n')
                score = detections[0, i, j, 0]
                label_name = labelmap[i-1]
                
                print(label_name)
                
                display_txt = '%s: %.2f'%(label_name, score)
                # display_txt = ' %.2f'%(score)
                
                pt = (detections[0, i, j, 1:]*scale).cpu().numpy()
                coords = (pt[0], pt[1], pt[2], pt[3])
                
                #color = colors[i]
#
                currentAxis.add_patch(patches.Rectangle((pt[0], pt[1]),pt[2]-pt[0], pt[3]-pt[1],
                                                         linewidth=2,edgecolor='b',facecolor='none'))
                currentAxis.text(pt[0], pt[1], display_txt, bbox={'facecolor':'b', 'alpha':0.5})
                
                pred_num += 1
                with open(filename, mode='a') as f:
                    f.write(str(pred_num)+' score: ' +
                            str(score) + ' '+' || '.join(str(c) for c in coords) + '\n')
                j += 1
#         plt.show()
        plt.savefig('./eval/'+img_id+'box.jpg',format='jpeg')
        plt.close()


def test_voc():
    # load net
    num_classes = len(VOC_CLASSES) + 1 # +1 background
    net = build_ssd('test', 300, num_classes) # initialize SSD
    net.load_state_dict(torch.load(args.trained_model, map_location=lambda storage, loc: storage))
    net.eval()
    print('Finished loading model!')
    # load data
    testset = VOCDetection(args.voc_root, [('2007', 'test')], None, VOCAnnotationTransform())
    if args.cuda:
        net = net.cuda()
        cudnn.benchmark = True
    # evaluation
    test_net(args.save_folder, net, args.cuda, testset,
             BaseTransform(net.size, (104, 117, 123)),
             thresh=args.visual_threshold)

if __name__ == '__main__':
    test_voc()
