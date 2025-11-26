# -*- coding: utf-8 -*-

import numpy as np
import h5py

from numba import jit, prange
from skimage.morphology import dilation, erosion, disk


def calculate_grid_pos(img_shape, batch_size=[128,]*3, origin=[0,0,0]):
    """
    Calculates the grid positions based on image size and the user-defined level block size

    Input:
        img: input image array
        batch_size: block size of x, y, z
    
    Output:
        List of block information for this level
        [0, 1]: x start and end indices
        [2, 3]: y start and end indices
        [4, 5]: z start and end indices
        [6, 7, 8]: total number of blocks across x, y, z
        
    """
    if not isinstance(img_shape, tuple):
        img_shape = img_shape.shape
    

    x_start = np.arange(origin[0], img_shape[0], batch_size[0])
    x_end = np.concatenate([x_start[1:], [img_shape[0]]])

    y_start = np.arange(origin[1], img_shape[1], batch_size[1])
    y_end = np.concatenate([y_start[1:], [img_shape[1]]])
    
    z_start = np.arange(origin[2], img_shape[2], batch_size[2])
    z_end = np.concatenate([z_start[1:], [img_shape[2]]])

    x_start_final = np.tile(x_start, len(y_start)*len(z_start))
    x_end_final = np.tile(x_end, len(y_end)*len(z_end))
    
    y_start_final = np.tile(np.repeat(y_start, len(x_start)), len(z_start))
    y_end_final = np.tile(np.repeat(y_end, len(x_end)), len(z_end))
    
    z_start_final = np.repeat(z_start, len(x_start)*len(y_start))
    z_end_final = np.repeat(z_end, len(x_end)*len(y_end))
    
    
    
    
    return [x_start_final, x_end_final, y_start_final, y_end_final, z_start_final, z_end_final, len(x_start), len(y_start), len(z_start)]


def calculate_block_idx(block_coords, cur_block_shape):
    cur_block_shape = block_coords[6:]
    
    # calculate the indices for each block
    x_size, y_size, z_size = cur_block_shape
    x_idx = np.tile(np.arange(0, x_size), y_size*z_size)
    y_idx = np.tile(np.repeat(np.arange(0, y_size), x_size), z_size)
    z_idx = np.repeat(np.arange(0, z_size), x_size*y_size)
    
    cur_total_idx = np.array([x_idx, y_idx, z_idx]).transpose()
    
    return cur_total_idx



def rename_duplicate_channels(channels):
    for i, ch in enumerate(channels):
        if channels.count(ch) > 1:
            dup_idx = [ i for i,x in enumerate(channels) if x == ch]
            for i, d in enumerate(dup_idx):
                if i > 0:
                    channels[d] = f'{ch}_{i}'
                    
    return channels


def get_channel_names(channels_input):
    
    ch_names = [None] * len(channels_input)
    
    for c in range(len(channels_input)):
        img_path, ch = channels_input[c]
        #print(img_path, ch)
        with h5py.File(img_path, 'r') as img_file:
            if f'DataSetInfo/Channel {ch}' in img_file.keys():
                ch_name = img_file[f'DataSetInfo/Channel {ch}'].attrs['Name']
                ch_name = [ str(s, encoding='UTF-8') for s in ch_name ]
                ch_names[c] = ''.join(ch_name)
            else:
                ch_names[c] = img_path.split('\\')[-1]
            
    return ch_names


def get_dimensions(img_path):
    
    dims = ['Z', 'Y', 'X']
    
    img_dimension = np.zeros((3), dtype=int)
    vxl_dimension = np.zeros((3), dtype=float)
    with h5py.File(img_path[0], 'r') as img_file:
        for n, dim in enumerate(dims):
            cur_dim = img_file['DataSetInfo/Image'].attrs[f'{dim}']
            cur_dim = ''.join(str(s, encoding='UTF-8') for s in cur_dim)
            img_dimension[n] = int(cur_dim)
        
            cur_vxl_max = img_file['DataSetInfo/Image'].attrs[f'ExtMax{2-n}']
            cur_vxl_max = ''.join(str(s, encoding='UTF-8') for s in cur_vxl_max)
            
            cur_vxl_min = img_file['DataSetInfo/Image'].attrs[f'ExtMin{2-n}']
            cur_vxl_min = ''.join(str(s, encoding='UTF-8') for s in cur_vxl_min)
            
            vxl_dimension[n] = (float(cur_vxl_max) - float(cur_vxl_min)) / float(cur_dim)
        
    return img_dimension, vxl_dimension


def load_image_block(channels_input, origin, blk_origin):

    max_ext = blk_origin[:,1]-blk_origin[:,0]

    img_block = np.zeros((len(channels_input), max_ext[0], max_ext[1], max_ext[2]), dtype=np.int64)

    origin_pads = np.array([0, 0, 0], dtype=np.int64)

    for c in range(len(channels_input)):
        
        ims_file, ch = channels_input[c]
        
        blk_start = origin[:,0] + blk_origin[:,0]
        blk_end = origin[:,0] + blk_origin[:,1]
        
        with h5py.File(ims_file, 'r') as fimg:
            img = fimg[f'DataSet/ResolutionLevel 0/TimePoint 0/Channel {ch}/Data']
            #print(img.shape)
            temp_image = img[blk_start[0]:blk_end[0],
                             blk_start[1]:blk_end[1],
                             blk_start[2]:blk_end[2]]
            
            temp_pads = []
            for ndim in range(3):
                # determine padding (origin_pad, difference between block size and temp_image shape minus origin_pad)
                temp_pads.append(tuple((origin_pads[ndim], max_ext[ndim]-temp_image.shape[ndim]-origin_pads[ndim])))
            temp_image = np.pad(temp_image, temp_pads)
            
        img_block[c] = temp_image[...]
        
    return img_block


@jit(nopython=True, parallel=True)
def fill_label_coords_id(label_coords, masks):
    label_id = np.zeros((len(label_coords)), dtype=np.int64)
                        
    for j in prange(len(label_coords)):
        label_id[j] = masks[label_coords[j,0],
                            label_coords[j,1],
                            label_coords[j,2]]
        
    return label_id





def extract_values_by_channels(im1_block, im2_block, channels=None, radial_params=None, mid_slice=False):
    
    mask_channels, context_channels = channels
    
    val_c = np.zeros((len(mask_channels), len(context_channels)), dtype=float)
    val_m = np.zeros((len(mask_channels)), dtype=float)
    
    if mid_slice:
        mid_z = im1_block.shape[1] // 2
        im1_block = im1_block[:, mid_z, :, :]
        im2_block = im2_block[:, mid_z, :, :]
    
    for mc in mask_channels:
        val_m[mc] = np.sum(im1_block[mc] > (np.mean(im1_block[mc]) + np.std(im1_block[mc])))
        for cc in context_channels: 
            im1, out = func_multiply(im1_block, im2_block, im1_ch = mc, im2_ch = cc)
            
            val_c[mc,cc] = np.sum(out[...]) / np.sum(im1[...])
            
            
    return val_c, val_m


def func_dilation(img, footprint=6):
    out = np.zeros_like(img)
    for z in range(img.shape[0]):
        out[z] = dilation(img[z], footprint=disk(footprint))        
    return out



def func_erosion(img, footprint=3):
    out = np.zeros_like(img)
    for z in range(img.shape[0]):
        out[z] = erosion(img[z], footprint=disk(footprint))
        
    return out

                  
                                                                 
def func_subtraction(img1, img2):
    return img1-img2


def func_multiply(im1_block, im2_block, im1_ch=None, im2_ch=None):

    im1_arr = im1_block[im1_ch,...].astype(float)
    im2_arr = im2_block[im2_ch,...].astype(float)
    
    out = im1_arr * im2_arr
    
    return im1_arr, out