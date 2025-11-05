import os
import os.path as osp
from tqdm import tqdm
from fire import Fire
import json
import sys
from glob import glob
import cv2
import lmdb
from utils import generate_patch_image, process_bbox
from dex_ycb_toolkit.factory import get_dataset
from dex_ycb_toolkit.dex_ycb import _SUBJECTS, _SERIALS
import os.path as osp
import os

def generate_image(img_folder='/home/liyuan/DexYCB', data_dir = osp.join(osp.dirname(__file__), 'data'),task='train'):
    with open(osp.join(data_dir, 'dexycb_{}_s0.json').format(task), 'r') as f:
         json_data = json.load(f)
    anno_data = json_data['annotations']
    img_data = json_data['images']
    image_data = osp.join(data_dir, 'images', task)
    if not osp.exists(image_data):
          os.makedirs(image_data)
    for idx in tqdm(range(len(img_data))):
          key = img_data[idx]['file_name']
          key_byte = key.encode('ascii')
          subject_id = _SUBJECTS[int(img_data[idx]['file_name'].split('_')[0]) - 1]
          video_id = '_'.join(img_data[idx]['file_name'].split('_')[1:3])
          cam_id = img_data[idx]['file_name'].split('_')[-2]
          frame_id = img_data[idx]['file_name'].split('_')[-1].rjust(6, '0')
          img_path = os.path.join(img_folder, subject_id, video_id, cam_id, 'color_' + frame_id + '.jpg')
          bbox = anno_data[idx]['bbox']   
          original_img_data = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
          data, _ = generate_patch_image(original_img_data, bbox, (256, 256), 1, 0)
          image_name = key+'.png'
          image_path = osp.join(image_data, image_name)
          cv2.imwrite(image_path, data)

if __name__ == '__main__':
      generate_image(task='train')
      generate_image(task='test')