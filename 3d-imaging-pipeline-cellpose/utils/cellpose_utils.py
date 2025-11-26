# -*- coding: utf-8 -*-

import numpy as np
from math import ceil

from numba import jit, prange

import h5py
import yaml


def read_yaml(file_path):
    with open(file_path, "r") as f:
        return yaml.safe_load(f)

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




def get_ims_block(i, input_ims, img_shape, coord_pos, origin, overlap, **kwargs):
    
    origin_pos = origin[:,0]
    
    start_pos = [coord_pos[0][i], coord_pos[2][i], coord_pos[4][i]] #+ origin_pos
    end_pos = [coord_pos[1][i], coord_pos[3][i], coord_pos[5][i]] #+ origin_pos

    start = [0, 0, 0]
    end = [0, 0, 0]
    

    for n in range(3):
        if start_pos[n] > 0:
            if start_pos[n]-overlap[n] > 0:
                start[n] = start_pos[n]-overlap[n]
        else:
            start[n] = 0

        if end_pos[n] > img_shape[n]:
            end[n] = img_shape[n]
        else:
            if end_pos[n]+overlap[n] < img_shape[n]:
                end[n] = end_pos[n]+overlap[n]
            else:
                end[n] = img_shape[n]
    
    
    crop_range = []
    for n in range(3):
        crop_range.append(slice(start[n]+origin_pos[n], end[n]+origin_pos[n]))
    
    crop_range = np.s_[tuple(crop_range)]
    
    
    if kwargs['channel']:
        ch = kwargs['channel']
    else:
        ch = 0
    
    if isinstance(ch, int):
        ch = [ch]
    
    if kwargs['res_lvl']:
        res_lvl = kwargs['res_lvl']
    else:
        res_lvl = 0

    output_block = []
        
    for c in ch:
        with h5py.File(input_ims, 'r') as f:
            data = f[f'DataSet/ResolutionLevel {res_lvl}/TimePoint 0/Channel {c}/Data']
            #print(data.shape)
            output_block.append(np.array(data[crop_range]))

    output_coords = np.array([start, end]).T

    return output_block, output_coords


def get_block(i, input_data, coord_pos, overlap):
    
    img_shape = input_data.shape
    
    start_pos = [coord_pos[0][i], coord_pos[2][i], coord_pos[4][i]]
    end_pos = [coord_pos[1][i], coord_pos[3][i], coord_pos[5][i]]

    start = [0, 0, 0]
    end = [0, 0, 0]
    
    for n in range(3):
        if start_pos[n] > 0:
            if start_pos[n]-overlap[n] > 0:
                start[n] = start_pos[n]-overlap[n]
        else:
            start[n] = 0

        if end_pos[n] > img_shape[n]:
            end[n] = img_shape[n]
        else:
            if end_pos[n]+overlap[n] < img_shape[n]:
                end[n] = end_pos[n]+overlap[n]
            else:
                end[n] = img_shape[n]
                
    output_block = input_data[start[0]:end[0],
                              start[1]:end[1],
                              start[2]:end[2]]
    

    output_coords = np.array([start, end]).T

    return output_block, output_coords


def calculate_block_dims(img_shape, block_size):
    
    dims = []
    
    print(img_shape)
    print(block_size)
    
    for d in range(3):
        dims.append(int(ceil(img_shape[d] / block_size[d])))
        
    return np.array(dims)
    
def find_cull_directions(i, total_idx, block_dims):
    cur_blk_idx = total_idx[i]
    dirs = [0]*6
    
    for d in range(3):
        if (cur_blk_idx[d] > 0) and (cur_blk_idx[d] <= block_dims[d]-1):
            dirs[d*2] = 1          
            
        if cur_blk_idx[d] < block_dims[d]-1:
            dirs[(d*2)+1] = 1
            
    return np.array(dirs)
    

@jit(nopython=True, parallel=True)
def find_bounding_box_fast(cur_blk, coord_corr, cull_directions, cull_switch=np.array([1,1,1])):
    
    labels_to_process = np.unique(cur_blk)
    labels_local = np.arange(len(labels_to_process))
    
    bbox = np.zeros((len(labels_to_process), 6), dtype=np.int32)
    
    
    label_coords = np.where(cur_blk > 0)
    label_coords = np.stack(label_coords, axis=1).astype(np.int32)
    
    
    label_values = np.zeros((len(label_coords)), dtype=np.int32)
    
    cull_state = np.zeros((len(labels_to_process)), dtype=np.int32)
    
    for j in prange(len(label_coords)):
        label_values[j] = cur_blk[label_coords[j,0],
                                  label_coords[j,1],
                                  label_coords[j,2]]


    for i in prange(len(labels_to_process)):

        label = labels_to_process[i]
        
        if label == 0: ## remove mask=0
            cull_state[i] = 1
        
        else:
            cur_lc = label_coords[label_values == i]
            
            for d in range(3):
                bbox[i, d*2] = np.min(cur_lc[:,d])
                bbox[i, (d*2)+1] = np.max(cur_lc[:,d])
            
            ## checking dims for culling
            for d in range(3):
                if cull_switch[d] > 0: ## check each dim if the switch for culling is on
                    if cull_directions[d*2] == 1:
                        if (bbox[i, d*2] <= 0):
                            cull_state[i] = 1
                    if cull_directions[(d*2)+1] == 1:
                        if (bbox[i, (d*2)+1] >= (cur_blk.shape[d]-1)):
                            cull_state[i] = 1

    
    return bbox, labels_to_process, cull_state




@jit(nopython=True)
def find_bounding_box(cur_blk, coord_corr, cull_directions, cull_switch=np.array([1,1,1])):
    
    labels_to_process = np.unique(cur_blk)
    labels_local = np.arange(len(labels_to_process))
    
    bbox = np.zeros((len(labels_to_process), 6), dtype=np.int32)
    to_cull = []
    
    for i, label in enumerate(labels_to_process):
        
        if label == 0: ## remove mask=0
            to_cull.append(i)
        
        else:
            label_coords = np.where(cur_blk == label)
            bounds = []
            for d in range(3):
                bounds.append(np.min(label_coords[d]))
                bounds.append(np.max(label_coords[d]))
            bbox[i] = np.array(bounds)
            
 
            ## checking dims for culling
            for d in range(3):
                if cull_switch[d] > 0: ## check each dim if the switch for culling is on
                    if cull_directions[d*2] == 1:
                        if (bbox[i, d*2] <= 0):
                            to_cull.append(i)
                    if cull_directions[(d*2)+1] == 1:
                        if (bbox[i, (d*2)+1] >= (cur_blk.shape[d]-1)):
                            to_cull.append(i)
    
    ## coordinate correction
    
    bbox += coord_corr
    
    ## new bbox
    
    labels_filtered = np.array([ x for x in labels_local if not x in to_cull ], dtype=np.int32)
    bbox_survived = bbox[labels_filtered]
    labels_survived = labels_to_process[labels_filtered]

    return bbox_survived, labels_survived