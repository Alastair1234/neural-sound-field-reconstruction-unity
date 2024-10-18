import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import os
import json
import random


def calculate_pixel_coords_from_corners(tl, tr, bl, br, h, w):
  # tl, tr, bl, br are all [3] tensors specifying the corner locations of the grid
  # h, w are integers specifying the number of pixels along each axis
  # return [h, w, 3] specifying the pixel locations of the grid
  axis_height = bl - tl
  axis_width = tr - tl
  axis_height_inc = axis_height / h
  axis_width_inc = axis_width / w
  grid = np.zeros([h, w, 3], dtype=float)
  for i in range(h):
    for j in range(w):
      grid[i, j, :] = tl + axis_height_inc * (i + 0.5) + axis_width_inc * (j + 0.5)
  return grid


class Unity_Dataset(Dataset):
  def __init__(self, directory, partition, h, w, isotropic=False, train_split_size=0.8):
    meta_filepath = os.path.join(directory, 'imageInfo.json')
    with open(meta_filepath, 'r') as f:
      meta_json = json.load(f)
    self.meta = {}
    self.filenames = []
    for json_info in meta_json['PointInfos']:
      filename = json_info['FileName']
      self.filenames.append(filename)
      select_info = {}
      self.meta[filename] = select_info

      image_tl_x = json_info['TopLeftCoordinate']['x']
      image_tl_y = json_info['TopLeftCoordinate']['y']
      image_tl_z = json_info['TopLeftCoordinate']['z']
      image_tl_coord = np.array([image_tl_x, image_tl_y, image_tl_z], dtype=float)
      select_info['image_tl_coord'] = image_tl_coord

      image_tr_x = json_info['TopRightCoordinate']['x']
      image_tr_y = json_info['TopRightCoordinate']['y']
      image_tr_z = json_info['TopRightCoordinate']['z']
      image_tr_coord = np.array([image_tr_x, image_tr_y, image_tr_z], dtype=float)
      select_info['image_tr_coord'] = image_tr_coord

      image_bl_x = json_info['BottomLeftCoordinate']['x']
      image_bl_y = json_info['BottomLeftCoordinate']['y']
      image_bl_z = json_info['BottomLeftCoordinate']['z']
      image_bl_coord = np.array([image_bl_x, image_bl_y, image_bl_z], dtype=float)
      select_info['image_bl_coord'] = image_bl_coord

      image_br_x = json_info['BottomRightCoordinate']['x']
      image_br_y = json_info['BottomRightCoordinate']['y']
      image_br_z = json_info['BottomRightCoordinate']['z']
      image_br_coord = np.array([image_br_x, image_br_y, image_br_z], dtype=float)
      select_info['image_br_coord'] = image_br_coord

      if not isotropic:
        sensor_x = json_info['Position']['x']
        sensor_y = json_info['Position']['y']
        sensor_z = json_info['Position']['z']
        sensor_coords = np.array([sensor_x, sensor_y, sensor_z], dtype=float)
        select_info['sensor_coords'] = sensor_coords
    self.partition = partition
    self.isotropic = isotropic
    self.h = h
    self.w = w
    self.directory = directory

    self.filenames = sorted(self.filenames, key=lambda x:random.random())
    cutoff = int(train_split_size * len(self.filenames))
    if self.partition == 'train':
      self.filenames = self.filenames[:cutoff]
    elif self.partition == 'test':
      self.filenames = self.filenames[cutoff:]

  def __len__(self):
    return len(self.filenames)

  def __getitem__(self, index):
    datapoint = {}
    filename = self.filenames[index]
    select_info = self.meta[filename]
    image_filepath = os.path.join(self.directory, filename)
    img = Image.open(image_filepath)
    img = img.convert('L')
    img = img.resize((self.w, self.h), Image.NEAREST)
    img = np.array(img, dtype=np.float32)
    datapoint['imgs'] = img
    datapoint['image_tl_coord'] = select_info['image_tl_coord']
    datapoint['image_tr_coord'] = select_info['image_tr_coord']
    datapoint['image_bl_coord'] = select_info['image_bl_coord']
    datapoint['image_br_coord'] = select_info['image_br_coord']
    datapoint['pixel_coords'] = calculate_pixel_coords_from_corners(datapoint['image_tl_coord'],
                                                                   datapoint['image_tr_coord'],
                                                                   datapoint['image_bl_coord'],
                                                                   datapoint['image_br_coord'],
                                                                   self.h, self.w)
    if not self.isotropic:
      datapoint['sensor_coords'] = select_info['sensor_coords']
    datapoint['filenames'] = filename
    return datapoint
