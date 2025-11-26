# -*- coding: utf-8 -*-

import numpy as np

from cellpose import models, core
from utils.cellpose_utils import get_ims_block, find_bounding_box_fast, find_cull_directions
from utils.cluster_setup import create_gpu_cluster

from dask.distributed import Client, as_completed

from tqdm import tqdm


def cellpose_dask(blk_id, cellpose_args, model_args, eval_args, seg_args):

    total_idx = seg_args['total_idx']
    cull_switch = seg_args['cull_switch']
    compressor = seg_args['compressor']
    block_dims = seg_args['block_dims']
    zslice_correction = seg_args['zslice_correction']
 

    cur_blk, cur_coords = get_ims_block(blk_id, **cellpose_args)
    
    if len(cur_blk) == 1:
        cur_blk = cur_blk[0]
        eval_args['channels'] = [0,0]
        
    elif len(cur_blk) > 1:
        cur_blk = np.stack(cur_blk)
        cur_blk = np.swapaxes(cur_blk, 0, 1)
        eval_args['channels'] = [1,2]
        eval_args['channel_axis'] = 1
    
    
    use_GPU = core.use_gpu()
    print('>>> GPU activated? %d'%use_GPU)

    model = models.Cellpose(gpu=use_GPU, **model_args)
    
    masks, flows, styles, _ = model.eval(cur_blk, **eval_args)
     
    if np.max(masks) > 0:
        zgrp = cellpose_args['zgrps'][blk_id]
        zarr_mask = zgrp.create_dataset('mask', shape=masks.shape, chunks=masks.shape, dtype=masks.dtype, compressor=compressor)
        zarr_mask[...] = masks[...]
        
        coord_corr = np.repeat(cur_coords[:,0], 2).astype(np.int32)
        cull_directions = find_cull_directions(blk_id, total_idx, block_dims)
        
        bbox_survived, labels_survived, cull_state = find_bounding_box_fast(masks, coord_corr, cull_directions,
                                                                            cull_switch = cull_switch)
        
        bbox_survived += coord_corr
        
        cull_state = np.invert(cull_state.astype(bool))

        bbox_survived = bbox_survived[cull_state]
        labels_survived = labels_survived[cull_state]
        
        ## check for single z-slice mask
        if zslice_correction:
            for i in range(len(bbox_survived)):
                if bbox_survived[i,1] - bbox_survived[i,0] == 0:
                    bbox_survived[i,1] += 1
                    
        zarr_bbox = zgrp.create_dataset('bbox', shape=bbox_survived.shape, chunks=bbox_survived.shape, dtype=bbox_survived.dtype, compressor=compressor)
        zarr_bbox[...] = bbox_survived[...]
        
        zarr_labels = zgrp.create_dataset('labels', shape=labels_survived.shape, chunks=labels_survived.shape, dtype=labels_survived.dtype, compressor=compressor)
        zarr_labels[...] = labels_survived[...]
        
        zarr_blk_coords = zgrp.create_dataset('blk_coords', shape=cur_coords.shape, chunks=cur_coords.shape, dtype=cur_coords.dtype, compressor=compressor)
        zarr_blk_coords[...] = cur_coords[...]

    return blk_id


def initialize_segmentation(cellpose_args, model_args, eval_args, gpu_dask_config, seg_args):

    cluster_mode = seg_args['cluster_mode']
    total_idx = seg_args['total_idx']
    
    FRACTION_WORKERS = 0.5    

    CLUSTER_SIZE = gpu_dask_config['cluster_size']
    
    cluster=create_gpu_cluster(mode=cluster_mode, config=gpu_dask_config)
    
    client = Client(cluster)
    
    if cluster_mode == 'SLURM':
        print(cluster.job_script())
        cluster.scale(CLUSTER_SIZE)
    
    
    client.wait_for_workers(n_workers=round(CLUSTER_SIZE*FRACTION_WORKERS))
    
    futures_seg_tasks = list(np.arange(len(total_idx)))
    
    futures_seg = []
    
    if len(futures_seg_tasks) >= CLUSTER_SIZE:
        init_batch = CLUSTER_SIZE
    else:
        init_batch = len(futures_seg_tasks)
    
    for i in range(init_batch):
        
        tid = futures_seg_tasks.pop(0)
        
        f = client.submit(cellpose_dask,
                          tid, cellpose_args, model_args, eval_args, seg_args)
        
        futures_seg.append(f)
    
    futures_seg_seq = as_completed(futures_seg)
    
    for future in tqdm(futures_seg_seq, total=len(total_idx), desc='Cellpose segmentation'):
        fi = future.result()
        
        if len(futures_seg_tasks) > 0:
            tid = futures_seg_tasks.pop(0)
            
            f_new = client.submit(cellpose_dask,
                              tid, cellpose_args, model_args, eval_args, seg_args)
            
            futures_seg_seq.add(f_new)
    
    client.shutdown()
    
    return True