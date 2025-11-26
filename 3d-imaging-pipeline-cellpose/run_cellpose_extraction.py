# -*- coding: utf-8 -*-

import os
import sys

import h5py
import zarr


import numpy as np
import pandas as pd

from utils.cellpose_utils import read_yaml

from cellpose_extraction.extraction_node import initialize_extraction
from cellpose_extraction.extract_utils import *


def detect_channels(ims_files):
    total_chs = []
    for ims_file in ims_files:
        chs = []
        with h5py.File(ims_file, 'r') as fimg:
            img_keys = list(fimg['DataSet/ResolutionLevel 0/TimePoint 0/'].keys())
            for c in img_keys:
                chs.append(int(c.split(' ')[-1]))
        chs = np.sort(chs)
        total_chs.append(chs)       
    return total_chs
        


def run_cellpose_extraction(config_path):

    res_lvl=0    

    config = read_yaml(config_path)
    
    channels_input = {}
    
    RUN_EXTRACTION = config['RunExtraction']
    
    input_path = os.path.join(config['ProjectPath'], config['InputDir'])
    output_path = os.path.join(config['ProjectPath'], config['OutputDir'])
    
    channels = config['Channels']
    ims_files = [ os.path.join(input_path, config['InputImage'][x]) for x in range(len(config['InputImage'])) ]
    ch_name_type = config['ChannelNames']
    
    if channels == 'all':
        channels = detect_channels(ims_files)
    
    print('Unprocessed channels: ', channels)
    
    block_ch = list(np.arange(len([ x for y in channels for x in y])))  ## channel id for contexts_block
    cch = [ x for y in channels for x in y] ## channels from each image
    cimg = [ [i] * len(x) for i, x in enumerate(channels)]
    cimg = [ x for y in cimg for x in y ]  ## image_id from each image

    for c in block_ch:
        channels_input[c] = [ims_files[cimg[c]], cch[c]]
    
    ########
    
    #### get image dimensions and voxel dimensions
    img_dims, vxl_dims = get_dimensions(ims_files)
    

    ### get channel names
    if ch_name_type == 'auto':
        ch_names = get_channel_names(channels_input)
    else:
        if isinstance(ch_name_type, list):
            ch_names = ch_name_type
    
    print('Channels input: ', channels_input)
    print('Channel Names: ', ch_names)
    
    
    ## get image attributes
    with h5py.File(ims_files[0], 'r') as f:
        data = f[f'DataSet/ResolutionLevel {res_lvl}/TimePoint 0/Channel {channels[0][0]}/Data']
        full_shape = data.shape
        
    print('Full shape: ', full_shape)
    
    if config['InputCrop'] is False:
        origin = np.array([[0, full_shape[0]],
                           [0, full_shape[1]],
                           [0, full_shape[2]]])

    elif isinstance(config['InputCrop'], list):
        origin = np.array(config['InputCrop'])
    

    for d in range(3):
        if origin[d][0] < 0: origin[d][0] = 0
        if origin[d][1] < 0: origin[d][1] = full_shape[d]
    
    print('Origin: ', origin)
    
    block_size = config['BlockShape']
    overlap = config['Overlap']
    
    block_coords = calculate_grid_pos(full_shape, batch_size=block_size)

    total_idx = calculate_block_idx(block_coords, overlap)

    
    ###################
    ## segmentation zarr
    zarr_grp = zarr.open(os.path.join(output_path, config['PredictionFileName']), 'r')
    
    if RUN_EXTRACTION:
    
        ## output zarr
        
        out_grp = zarr.open(os.path.join(output_path, config['ExtractionFileName']), 'a')
        zgrps = []
        for i in range(len(total_idx)):
            zgrps.append(out_grp.create_group(f'block_{i}'))

        
        dask_config = { 'cluster_size'       : config['DASK']['EXTRACTION']['cluster_size'], 
                        'processes'          : config['DASK']['EXTRACTION']['processes'], 
                        'cores'              : config['DASK']['EXTRACTION']['cores'], 
                        'memory'             : config['DASK']['EXTRACTION']['memory'],
                        'walltime'           : config['DASK']['EXTRACTION']['walltime'], 
                        'cpu_type'           : config['DASK']['EXTRACTION']['cpu_type']}
    
        
        extract_args = { 'zarr_grp' : zarr_grp,
                         'out_grp' : out_grp,
                         'channels_input' : channels_input,
                         'origin' : origin,
                         'total_idx' : total_idx}
    
        
        _ = initialize_extraction(extract_args, dask_config, config)
    
    else:
        out_grp = zarr.open(os.path.join(output_path, config['ExtractionFileName']), 'r')

    ################################################################################
    
    print('Finalizing data extraction...')
    
    for i in range(len(total_idx)):
        ## get block info and final bounding box
        if 'labels' in zarr_grp[f'block_{i}'].keys(): ### 
            if not 'labels' in out_grp[f'block_{i}'].keys():
                print('Problem block: ', i)
    
    extracted_array = [[]] * len(total_idx)
    cell_coords = [[]] * len(total_idx)
    gidx_arr = [[]] * len(total_idx)
    blk_idx_arr = [[]] * len(total_idx)

    for i in range(len(total_idx)): 
        if 'labels' in zarr_grp[f'block_{i}'].keys(): ###          
            if 'labels' in out_grp[f'block_{i}'].keys():
                ## get block info and final bounding box
                ## create array with block number
                glbl = np.array(zarr_grp[f'block_{i}/labels'])
                cur_blk_idx = np.zeros((len(glbl)))
                cur_blk_idx[...] = i
                blk_idx_arr[i] = cur_blk_idx
                
                
                extracted_array[i] = np.array(out_grp[f'block_{i}/cell_values'])
                cell_coords[i] = np.array(out_grp[f'block_{i}/cell_coords'])
                gidx_arr[i] = np.array(out_grp[f'block_{i}/labels'])
            else:
                print(f'Problem block: {i}. Not extracted.')
            
    
    ### remove empty lists
    extracted_array = [ x for x in extracted_array if len(x) > 0]
    cell_coords = [ x for x in cell_coords if len(x) > 0]
    gidx_arr = [ x for x in gidx_arr if len(x) > 0]
    blk_idx_arr = [ x for x in blk_idx_arr if len(x) > 0]
    
    ### concatenate all other arrays
    extracted_array = np.concatenate(extracted_array, axis=0)
    cell_coords = np.concatenate(cell_coords, axis=0)
    gidx_arr = np.concatenate(gidx_arr, axis=0)
    blk_idx_arr = np.concatenate(blk_idx_arr, axis=0)
    
    blk_idx_arr = blk_idx_arr.reshape(-1,1)
    gidx_arr = gidx_arr.reshape(-1,1)
    
    combined_idx_arr = np.concatenate((gidx_arr, blk_idx_arr), axis=1)
    
    #############################################
    
    output_prefix = config['OutputFilePrefix']
    return_csv = config['OutputCSV']
    coords_type = config['CellCoordinates']
    vxl_type = config['VoxelDimensions']
    
    ## override voxel dimensions with user defined list
    if isinstance(vxl_type, list):
        if len(vxl_type) == 3:
            vxl_dims = np.array(vxl_type)[::-1].astype(float)  # reverse from xyz to zyx
            print(f'Voxel dimensions: user defined {vxl_dims[::-1]}')
    elif vxl_type == 'auto':
        print(f'Voxel dimensions: auto from image file {vxl_dims[::-1]}')
    else:
        vxl_dims = np.array([1.0, 1.0, 1.0]).astype(float)
        print(f'Voxel dimensions data not provided: using default {vxl_dims}')
    
    
    output_array_path = f'{output_path}/{output_prefix}'
    
    print('Converting data to h5')
    print(f'Output location: {output_array_path}')
    
    layers = ['segmented', 'cell', 'nuclear', 'membrane']  
    
    ## set attributes with h5py
    with h5py.File(f'{output_array_path}.h5', mode='w') as f:
        f.create_group('Info')
        f['Info'].attrs['Image Dimensions'] = img_dims
        f['Info'].attrs['Voxel Dimensions'] = vxl_dims
        f['Info'].attrs['Channel Names'] = ch_names
        
    ## export individual layers
    for i, layer in enumerate(layers):
        pd_arr = pd.DataFrame(extracted_array[:,i,:], columns=ch_names)
        pd_arr.to_hdf(f'{output_array_path}.h5', key=layer, mode='a')
        
        if return_csv:
            pd_arr.to_csv(f'{output_array_path}_{layer}.csv')

    ## export coordinates
    if coords_type == 'world':
        cell_coords *= vxl_dims
    
    coords_arr = pd.DataFrame(cell_coords, columns=['Z', 'Y', 'X'])
    coords_arr.to_hdf(f'{output_array_path}.h5', key='positions', mode='a')
    if return_csv:
        coords_arr.to_csv(f'{output_array_path}_positions.csv')
    
    idx_arr = pd.DataFrame(combined_idx_arr, columns=['Cell Index', 'Block Index'])
    idx_arr.to_hdf(f'{output_array_path}.h5', key='indices', mode='a')
    if return_csv:
        idx_arr.to_csv(f'{output_array_path}_indices.csv')
    
    
    print('End of script')
    
    
    
if __name__ == '__main__':
    config_path = sys.argv[1]
    run_cellpose_extraction(config_path)
