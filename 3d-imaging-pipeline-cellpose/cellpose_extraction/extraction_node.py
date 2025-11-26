# -*- coding: utf-8 -*-

import numpy as np

from numcodecs import Blosc
from tqdm import tqdm

from skimage.filters import gaussian

from utils.cluster_setup import create_cluster
from dask.distributed import Client, as_completed

from cellpose_extraction.extract_utils import load_image_block, fill_label_coords_id, func_dilation, func_erosion, func_subtraction, extract_values_by_channels

def extract_by_tile(blk, zarr_grp, out_grp, channels_input, origin, config):
    
    compressor = Blosc(cname=config['Compression'], clevel=config['CompressionLevel'], shuffle=Blosc.BITSHUFFLE)
    
    mask_dilation = config['MaskDilation']
    mask_erosion = config['MaskErosion']
    mask_sigma = config['MaskGaussianSigma']
    min_size = config['MinSize']
    mid_slice = config['MidSlice']
    
    zgrp = zarr_grp[f'block_{blk}']
    ogrp = out_grp[f'block_{blk}']
    
    if 'labels' in zgrp.keys():
        glbl_idx = np.array(zgrp['labels'])
        masks = np.array(zgrp['mask'])
        blk_origin = np.array(zgrp['blk_coords'])
        
        channel_len = len(channels_input) 
        
        img_block = load_image_block(channels_input, origin, blk_origin)
        cell_values_all, mask_values_all, cell_coords_all = get_cell_value_new(glbl_idx,
                                                                               masks,
                                                                               origin,
                                                                               blk_origin,
                                                                               img_block,
                                                                               channel_len,
                                                                               min_size=min_size,
                                                                               mask_sigma=mask_sigma,
                                                                               dilation_fp=mask_dilation,
                                                                               erosion_fp=mask_erosion,
                                                                               mid_slice=mid_slice)
        
        out_cv = ogrp.create_dataset('cell_values', shape=cell_values_all.shape, chunks=(1000,4,channel_len), dtype=cell_values_all.dtype, compressor=compressor)
        out_cv[...] = cell_values_all[...]
        
        out_mv = ogrp.create_dataset('mask_values', shape=mask_values_all.shape, chunks=(1000,4), dtype=mask_values_all.dtype, compressor=compressor)
        out_mv[...] = mask_values_all[...]
        
        out_cc = ogrp.create_dataset('cell_coords', shape=cell_coords_all.shape, chunks=(1000,3), dtype=cell_coords_all.dtype, compressor=compressor)
        out_cc[...] = cell_coords_all[...]
        
        out_gidx = ogrp.create_dataset('labels', shape=glbl_idx.shape, chunks=glbl_idx.shape, dtype=glbl_idx.dtype, compressor=compressor)
        out_gidx[...] = glbl_idx[...]
                      
        return blk
    
    else:
        print(f'Block not found: {blk}')
        return []



def get_cell_value_new(full_labels, masks, origin, blk_origin, img_block, channel_len,
                       min_size=np.array([1, 15, 15]),
                       mask_sigma=1.0,
                       dilation_fp=1,
                       erosion_fp=8,
                       mid_slice=False):
    
    cell_values_all = np.zeros((len(full_labels), 4, channel_len), dtype=float)
    mask_values_all = np.zeros((len(full_labels), 4), dtype=int)
    cell_coords_all = np.zeros((len(full_labels), 3), dtype=float)

    batch_size = 1000

    batch_idx = np.arange(0, len(full_labels), batch_size)
    if batch_idx[-1] != len(full_labels):
        batch_idx = np.append(batch_idx, len(full_labels))
    
    ##################
    
    full_label_coords = np.where(masks > 0)
    full_label_coords = np.stack(full_label_coords, axis=1)
    
    label_id = fill_label_coords_id(full_label_coords, masks)
    
    for bi in tqdm(range(len(batch_idx)-1)):
    
        labels_to_process = full_labels[batch_idx[bi]:batch_idx[bi+1]]    
    
        cell_values = np.zeros((len(labels_to_process), 4, channel_len), dtype=float)
        cell_coords = np.zeros((len(labels_to_process), 3), dtype=float)
        mask_values = np.zeros((len(labels_to_process), 4), dtype=int)
         
        for l in range(len(labels_to_process)):
            
            lbl = labels_to_process[l]
            
            label_coords = full_label_coords[label_id == lbl]
        
            coords_center = []
            extent = []
            for ndim in range(3):
                coords_center.append(np.median(label_coords[:,ndim]))
                extent.append(np.max(label_coords[:,ndim])-np.min(label_coords[:,ndim])+1)
            coords_center = np.array(coords_center).astype(np.int64)
            extent = np.array(extent)
            
            ## input cell coordinates
            c_coords = (origin[:,0] + blk_origin[:,0] + coords_center)
            
            ## check size
            size_check = np.sum(extent >= min_size)
            
            if size_check == 3:
                mask_block_size = np.ceil(extent * 2).astype(int)
                offsets = np.floor(mask_block_size * 0.5).astype(int)
                
                coords_arr2 = (label_coords - coords_center + offsets).astype(np.int64)
                
                mask_block = np.zeros((4, mask_block_size[0], mask_block_size[1], mask_block_size[2]), dtype=np.int64)
                mask_block[0, coords_arr2[:,0],
                              coords_arr2[:,1],
                              coords_arr2[:,2]] = 1
                
                mask_block[1,:,:,:] = func_dilation(mask_block[0], footprint=dilation_fp)
                mask_block[2,:,:,:] = func_erosion(mask_block[0], footprint=erosion_fp)
                mask_block[3,:,:,:] = func_subtraction(mask_block[1,:,:,:], mask_block[2,:,:,:])
                
                mask_block_gaus = np.zeros_like(mask_block, dtype=float)
                
                ## gaussian filtering masks
                if mask_sigma > 0:
                    for b in range(mask_block.shape[0]):
                        for z in range(mask_block.shape[1]):
                            mask_block_gaus[b, z, ...] = gaussian(mask_block[b][z], sigma=mask_sigma)
                else:
                    mask_block_gaus = mask_block
    
                ## extract from individual channel   
                context_block = np.zeros((img_block.shape[0], mask_block_size[0], mask_block_size[1], mask_block_size[2]), dtype=np.int64)
                
                coords_origin = (coords_center - offsets).astype(np.int64)
                coords_end = (coords_origin + mask_block_size).astype(np.int64)
                
                ## check if coords_origin is < 0, which will fail to extract the context
                origin_pads = np.array([0, 0, 0], dtype=np.int64)
                
                if np.sum(coords_origin < 0) > 0:
                    origin_pads[coords_origin < 0] = -coords_origin[coords_origin < 0]  ## origin pad
                    coords_origin[coords_origin < 0] = 0  ## set negative origin to 0
                
                for c in range(img_block.shape[0]):             
                    temp_image= img_block[c,
                                           coords_origin[0]:coords_end[0],
                                           coords_origin[1]:coords_end[1],
                                           coords_origin[2]:coords_end[2]]
              
                    temp_pads = []
                    for ndim in range(3):
                        # determine padding (origin_pad, difference between block size and temp_image shape minus origin_pad)
                        temp_pads.append(tuple((origin_pads[ndim], mask_block_size[ndim]-temp_image.shape[ndim]-origin_pads[ndim])))
                    temp_image = np.pad(temp_image, temp_pads)
                    
                    context_block[c, :, :, :] = temp_image[...]

                mask_channels = np.arange(mask_block_gaus.shape[0])
                context_channels = np.arange(context_block.shape[0])
                
                val_c, val_m = extract_values_by_channels(im1_block=mask_block_gaus,
                                                            im2_block=context_block,
                                                            channels=[mask_channels, context_channels],
                                                            radial_params = None,
                                                            mid_slice=mid_slice
                                                            )
                
                cell_values[l] = val_c
                mask_values[l] = val_m
                cell_coords[l] = c_coords
            
        cell_values_all[batch_idx[bi]:batch_idx[bi+1]] = cell_values
        mask_values_all[batch_idx[bi]:batch_idx[bi+1]] = mask_values
        cell_coords_all[batch_idx[bi]:batch_idx[bi+1]] = cell_coords
        
    return cell_values_all, mask_values_all, cell_coords_all


def initialize_extraction(extract_args, dask_config, config):
    
    FRACTION_WORKERS = 0.5
    
    zarr_grp = extract_args['zarr_grp']
    out_grp = extract_args['out_grp']
    channels_input = extract_args['channels_input']
    origin = extract_args['origin']
    total_idx = extract_args['total_idx']
    
    
    CLUSTER_SIZE = dask_config['cluster_size']
    cluster_mode = config['DASK']['cluster_mode']
    
    cluster=create_cluster(mode=cluster_mode, config=dask_config)
    
    client = Client(cluster)
    
    if cluster_mode == 'SLURM':
        print(cluster.job_script())
        cluster.scale(CLUSTER_SIZE)
    
    client.wait_for_workers(n_workers=round(CLUSTER_SIZE*FRACTION_WORKERS))
    
    futures_ext_tasks = list(np.arange(len(total_idx)))
    
    futures_ext = []
    
    if len(futures_ext_tasks) >= CLUSTER_SIZE:
        init_batch = CLUSTER_SIZE
    else:
        init_batch = len(futures_ext_tasks)
    
    for i in range(init_batch):
        
        tid = futures_ext_tasks.pop(0)
        
        f = client.submit(extract_by_tile, tid, zarr_grp, out_grp, channels_input, origin, config)
        
        futures_ext.append(f)
    
    futures_ext_seq = as_completed(futures_ext)
    
    for future in tqdm(futures_ext_seq, total=len(total_idx), desc='Cellpose extraction'):
        fi = future.result()
        
        if len(futures_ext_tasks) > 0:
            tid = futures_ext_tasks.pop(0)
            
            f_new = client.submit(extract_by_tile, tid, zarr_grp, out_grp, channels_input, origin, config)
            
            futures_ext_seq.add(f_new)
    
    client.shutdown()
    
    return True

