import os

import numpy
import torch
import json
import matplotlib.pyplot as plt
import h5py
from PIL import Image, ImageDraw, ImageFont

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

BATCH_SIZE = 32
QUES_LEN = 20
IMAGE_REGION = 36
IMAGE_ATTENTION_THRES = 0.4
QUESTION_ATTENTION_THRES = -3
IMAGE_ATTENTION_MIN = 0
QUESTION_ATTENTION_MIN = -3

ques_path = 'attention_map_vis/question.json'
with open(ques_path, 'r') as file:
    ques = json.load(file)
for item in ques[:BATCH_SIZE]:
    print(item['ques'])

base_ques_scores = torch.load('attention_map_vis/data/base_batch1_q_scores.npy', map_location=torch.device('cpu'))
base_ques_scores[base_ques_scores < QUESTION_ATTENTION_THRES] = QUESTION_ATTENTION_MIN
base_img_scores = torch.load('attention_map_vis/data/base_batch1_i_scores.npy', map_location=torch.device('cpu'))
base_img_scores[base_img_scores < IMAGE_ATTENTION_THRES] = IMAGE_ATTENTION_MIN

cfq_ques_scores = torch.load('attention_map_vis/data/cfq_batch1_q_scores.npy', map_location=torch.device('cpu'))
cfq_ques_scores[cfq_ques_scores < QUESTION_ATTENTION_THRES] = QUESTION_ATTENTION_MIN
cfq_img_scores = torch.load('attention_map_vis/data/cfq_batch1_i_scores.npy', map_location=torch.device('cpu'))
cfq_img_scores[cfq_img_scores < IMAGE_ATTENTION_THRES] = IMAGE_ATTENTION_MIN

cfi_ques_scores = torch.load('attention_map_vis/data/cfi_batch1_q_scores.npy', map_location=torch.device('cpu'))
cfi_ques_scores[cfi_ques_scores < QUESTION_ATTENTION_THRES] = QUESTION_ATTENTION_MIN
cfi_img_scores = torch.load('attention_map_vis/data/cfi_batch1_i_scores.npy', map_location=torch.device('cpu'))
cfi_img_scores[cfi_img_scores < IMAGE_ATTENTION_THRES] = IMAGE_ATTENTION_MIN

# visdial_val_features.h5 include the information related bounding boxes, obtained through pre-trained Faster R-CNN
with h5py.File('/home/data/visdial_v1.0_test-std/visdial_val_features.h5', 'r') as file:
    boxes = file['boxes'][:]
    h = file['h'][:]
    w = file['w'][:]
    _image_ids = list(map(int, file['image_id']))


def reshape_ques(x):
    x = (x.sum(dim=1)).sum(dim=2)
    x = x / x.sum(dim=-1, keepdim=True)
    x = x.reshape(BATCH_SIZE, -1, QUES_LEN)
    return x.cpu().numpy()


def reshape_img(x):
    x = x.reshape(BATCH_SIZE, -1, IMAGE_REGION, IMAGE_REGION)
    x = x.sum(dim=1)
    x = x.sum(dim=-1)
    return x.cpu().numpy()


base_ques_scores = reshape_ques(base_ques_scores)
base_img_scores = reshape_img(base_img_scores)

cfq_ques_scores = reshape_ques(cfq_ques_scores)
cfq_img_scores = reshape_img(cfq_img_scores)

cfi_ques_scores = reshape_ques(cfi_ques_scores)
cfi_img_scores = reshape_img(cfi_img_scores)


def save_subfig(fig, ax, save_path, fig_name):
    bbox = ax.get_tightbbox(fig.canvas.get_renderer())
    extent = bbox.transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(save_path + fig_name, bbox_inches=extent)


def visualize_ques(id, base, cfq, cfi, origin):
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))

    axes[0, 0].imshow(base, cmap='plasma', interpolation='nearest')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(cfq, cmap='plasma', interpolation='nearest')
    axes[0, 1].axis('off')

    axes[1, 0].imshow(cfi, cmap='plasma', interpolation='nearest')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(origin)
    axes[1, 1].axis('off')

    plt.tight_layout()
    save_path_dir = 'attention_map_vis/data/ques/'
    sub_save_path_dir = 'attention_map_vis/data/sub_ques/'
    plt.savefig(f'{save_path_dir}heatmaps_ques{id}.png', dpi=600)
    save_subfig(fig, axes[0, 0], sub_save_path_dir, f'base{id}.png')
    save_subfig(fig, axes[0, 1], sub_save_path_dir, f'cfq{id}.png')
    save_subfig(fig, axes[1, 0], sub_save_path_dir, f'cfi{id}.png')
    save_subfig(fig, axes[1, 1], sub_save_path_dir, f'origin{id}.png')
    # plt.show()


def visualize_img(id, base, cfq, cfi, origin=None):
    if origin is None:
        origin = base
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))

    axes[0, 0].imshow(base, cmap='plasma', interpolation='nearest')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(cfq, cmap='plasma', interpolation='nearest')
    axes[0, 1].axis('off')

    axes[1, 0].imshow(cfi, cmap='plasma', interpolation='nearest')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(origin)
    axes[1, 1].axis('off')

    plt.tight_layout()
    save_path_dir = 'attention_map_vis/data/imgs/'
    sub_save_path_dir = 'attention_map_vis/data/sub_imgs/'
    plt.savefig(f'{save_path_dir}heatmaps_img{id}.png', dpi=600)
    save_subfig(fig, axes[0, 0], sub_save_path_dir, f'base{id}.png')
    save_subfig(fig, axes[0, 1], sub_save_path_dir, f'cfq{id}.png')
    save_subfig(fig, axes[1, 0], sub_save_path_dir, f'cfi{id}.png')
    save_subfig(fig, axes[1, 1], sub_save_path_dir, f'origin{id}.png')
    # plt.show()


def acc_img_attention(img_id, scores):
    index = _image_ids.index(img_id)

    box = boxes[index]
    hh = h[index]
    ww = w[index]

    print(hh, ww)
    ts = numpy.zeros([hh, ww])
    for r in range(IMAGE_REGION):
        x1, y1, x2, y2 = box[r]
        y1, y2 = int(y1), int(y2) - 1
        x1, x2 = int(x1), int(x2) - 1
        if y1 >= y2 or x1 >= x2:
            continue
        try:
            ts[int(y1):int(y2) - 1, int(x1) + 1:int(x2) - 1] += scores[r]
        except:
            pass
    return ts


val_img_path = '/home/data/visdial_v1.0_test-std/VisualDialog_val2018/'
for i in range(BATCH_SIZE):
    _base_ques_scores = base_ques_scores[i]
    _cfq_ques_scores = cfq_ques_scores[i]
    _cfi_ques_scores = cfi_ques_scores[i]

    width = 1600
    height = 1000
    image = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("attention_map_vis/UbuntuMono-R.ttf", 80)
    text_width, text_height = draw.textsize(ques[i]['ques'], font)
    y = (height - text_height) // 2
    draw.text((0, y), ques[i]['ques'], fill='black', font=font)
    image.save('attention_map_vis/tmp.png')
    visualize_ques(i, _base_ques_scores, _cfq_ques_scores, _cfi_ques_scores, numpy.array(
        Image.open(
            'attention_map_vis/tmp.png')))

    img_id = ques[i]['id']
    _base_img_scores = base_img_scores[i]
    _cfq_img_scores = cfq_img_scores[i]
    _cfi_img_scores = cfi_img_scores[i]
    base_acc = acc_img_attention(img_id, _base_img_scores)
    cfq_acc = acc_img_attention(img_id, _cfq_img_scores)
    cfi_acc = acc_img_attention(img_id, _cfi_img_scores)

    visualize_img(i, base_acc, cfq_acc, cfi_acc, numpy.array(
        Image.open(
            f'{val_img_path}VisualDialog_val2018_000000{str(img_id).zfill(6)}.jpg')))
