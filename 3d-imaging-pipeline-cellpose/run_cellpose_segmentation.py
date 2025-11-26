# -*- coding: utf-8 -*-

import numpy as np
import os
import sys

import zarr
from numcodecs import Blosc

import h5py

from utils.cellpose_utils import *
from cellpose_segmentation.segmentation_node import initialize_segmentation
from cellpose_segmentation.overlap_node import remove_overlaps


####################################

def run(config_path):
    
    print('initialize_segmentation: ', initialize_segmentation)
    
    res_lvl = 0
    
    config = read_yaml(config_path)
    
    print(config)
    RUN_PREDICTION = config['RunPrediction']
    RUN_REMOVE_OVERLAPS = config['RunOverlapRemoval']
    
    model_args = {}

    if config['CELLPOSE_MODEL']['ModelType'] == 'library':
        model_args['model_type'] = config['CELLPOSE_MODEL']['ModelName']
    elif config['CELLPOSE_MODEL']['ModelType'] == 'custom':
        model_args['pretrained_model'] = os.path.join(config['ProjectPath'], config['CELLPOSE_MODEL']['ModelDir'])
    
    
    eval_args = config['CELLPOSE_EVAL']
    
    print('Cellpose model: ', model_args)
    print('Cellpose eval: ', eval_args)
    
    block_size = config['BlockShape']
    overlap = config['Overlap']
    
    cluster_mode = config['DASK']['cluster_mode']
    
    output_path = os.path.join(config['ProjectPath'], config['OutputDir'])
    input_path = os.path.join(config['ProjectPath'], config['InputDir'])
    
    
    input_ims = os.path.join(input_path, config['InputImage'])
    
    gpu_dask_config = config['DASK']['Prediction']

    ###########################
    
    ch = config['InputChannels']
    
    if isinstance(ch, list):
        ch_init = int(ch[0])
    else:
        ch_init = int(ch)
    
    with h5py.File(input_ims, 'r') as f:
        data = f[f'DataSet/ResolutionLevel {res_lvl}/TimePoint 0/Channel {ch_init}/Data']
        full_shape = data.shape
    
    if config['InputCrop'] is False:
        origin = np.array([[0, full_shape[0]],
                           [0, full_shape[1]],
                           [0, full_shape[2]]])

    elif isinstance(config['InputCrop'], list):
        origin = np.array(config['InputCrop'])

    
    cull_switch = np.array(config['cull_switch'], dtype=np.uint8)
    zslice_correction = config['zslice_correction']
    
    
    #########
    
    print('Full shape: ', full_shape)
    
    #### set crop region
    
    for d in range(3):
        if origin[d][0] < 0: origin[d][0] = 0
        if origin[d][1] < 0: origin[d][1] = full_shape[d]
    
    print('Crop region: ', origin)
    
    ### get the cropped image shape
    
    img_shape = tuple(origin[:,1] - origin[:,0])
    
    print('Img shape (cropped): ', img_shape)
    
    
    block_dims = calculate_block_dims(img_shape, block_size)
    
    ## get the output image shape
    
    out_shape = tuple(block_size * block_dims)
    
    block_coords = calculate_grid_pos(out_shape, batch_size=block_size)
    total_idx = calculate_block_idx(block_coords, overlap)
    
    print('Total blocks: ', len(total_idx))
    
    cellpose_args = {'input_ims' : input_ims,
                     'img_shape' : out_shape,
                     'coord_pos' : block_coords,
                     'origin' : origin,
                     'overlap' : overlap,
                     'channel' : ch,
                     'res_lvl' : res_lvl}
    
    
    print(cellpose_args)
    
    zarr_output_path = os.path.join(output_path, config['PredictionFileName'])
    
    #####################################################################
    
    if RUN_PREDICTION:
        print('Running prediction')
        
        zarr_grp = zarr.open(zarr_output_path, mode='w')
        zgrps = []
        for i in range(len(total_idx)):
            zgrps.append(zarr_grp.create_group(f'block_{i}'))
        
        compressor = Blosc(cname=config['Compression'], clevel=config['CompressionLevel'], shuffle=Blosc.BITSHUFFLE)
        
        print('Zarr file created at: ', zarr_output_path)
        
        cellpose_args['zgrps'] = zgrps
        
        seg_args = {#'script_path' : script_path,
                    'cluster_mode' : cluster_mode,
                    'total_idx' : total_idx,
                    'cull_switch' : cull_switch,
                    'compressor' : compressor,
                    'block_dims' : block_dims,
                    'zslice_correction' : zslice_correction}
        
        seg_state = initialize_segmentation(cellpose_args, model_args, eval_args, gpu_dask_config,
                                            seg_args)
        
        print(seg_state)
    
    else:
        print('Prediction step skipped - loading segmentation data from zarr')
        zarr_grp = zarr.open(zarr_output_path, mode='a')
        
    #####################################################################

    if RUN_REMOVE_OVERLAPS:
        print('Running overlap removal')
        
        blocks_bbox, blocks_labels = remove_overlaps(total_idx, block_dims, overlap, full_shape, zarr_grp)
        
        for i in range(len(total_idx)):
            if len(blocks_labels[i]) > 0:
                zarr_grp[f'block_{i}']['bbox'] = blocks_bbox[i][...]
                zarr_grp[f'block_{i}']['labels'] = blocks_labels[i][...]
    
        
    print('End of script')
    
if __name__ == '__main__':
    config_path = sys.argv[1]
    run(config_path)