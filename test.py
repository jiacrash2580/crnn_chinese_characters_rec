import numpy as np
import sys, os
import time

sys.path.append(os.getcwd())

# crnn packages
import torch
from torch.autograd import Variable
import utils
import dataset
from PIL import Image
import models.crnn as crnn
import alphabets
str1 = alphabets.alphabet

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--images_path', type=str, default='test_images/test1.png', help='the path to your images')
opt = parser.parse_args()

# crnn params
# 3p6m_third_ac97p8.pth
crnn_model_path = 'trained_models/mixed_second_finetune_acc97p7.pth'
alphabet = str1
nclass = len(alphabet)+1

# crnn文本信息识别
def crnn_recognition(cropped_image, model):
    converter = utils.strLabelConverter(alphabet)
    image = cropped_image.convert('L')
    ##
    w = int(image.size[0] / (280 * 1.0 / 160))
    transformer = dataset.resizeNormalize((w, 32))
    image = transformer(image)
    if torch.cuda.is_available():
        image = image.cuda()
    image = image.view(1, *image.size())
    image = Variable(image)

    model.eval()
    preds = model(image)

    _, preds = preds.max(2)
    preds = preds.transpose(1, 0).contiguous().view(-1)

    preds_size = Variable(torch.IntTensor([preds.size(0)]))
    sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
    print('results: {0}'.format(sim_pred))

if __name__ == '__main__':
	# crnn network
    model = crnn.CRNN(32, 1, nclass, 256)
    print('loading pretrained model from {0}'.format(crnn_model_path))
    if torch.cuda.is_available():
        model = model.cuda()
        # 导入已经训练好的crnn模型,gpu模式加载
        model.load_state_dict(torch.load(crnn_model_path))
    else:
        # 模型是gpu模型,通过cpu模式加载
        model.load_state_dict(torch.load(crnn_model_path, map_location='cpu'))
    started = time.time()
    ## read an image
    image = Image.open(opt.images_path)

    crnn_recognition(image, model)
    finished = time.time()
    print('elapsed time: {0}'.format(finished-started))