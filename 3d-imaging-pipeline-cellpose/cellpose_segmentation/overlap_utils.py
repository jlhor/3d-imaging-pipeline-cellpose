# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 15:34:12 2025

@author: horj2
"""


import numpy as np
from numba import jit

## from cellpose.metrics
@jit(nopython=True)
def _label_overlap(x, y):
    """Fast function to get pixel overlaps between masks in x and y.

    Args:
        x (np.ndarray, int): Where 0=NO masks; 1,2... are mask labels.
        y (np.ndarray, int): Where 0=NO masks; 1,2... are mask labels.

    Returns:
        overlap (np.ndarray, int): Matrix of pixel overlaps of size [x.max()+1, y.max()+1].
    """
    # put label arrays into standard form then flatten them
    #     x = (utils.format_labels(x)).ravel()
    #     y = (utils.format_labels(y)).ravel()
    x = x.ravel()
    y = y.ravel()

    # preallocate a "contact map" matrix
    overlap = np.zeros((1 + x.max(), 1 + y.max()), dtype=np.uint)

    # loop over the labels in x and add to the corresponding
    # overlap entry. If label A in x and label B in y share P
    # pixels, then the resulting overlap is P
    # len(x)=len(y), the number of pixels in the whole image
    for i in range(len(x)):
        overlap[x[i], y[i]] += 1
    return overlap


### determine neighboring block

def find_adjacent_blocks(total_idx, block_dims):
    
    all_edges = []  ## first tile idx, second tile idx, direction
    all_dirs = [] ## cull directions for each current block
    all_adj_blks = [] ## idx of adjacent block for each current block
    
    for i in range(len(total_idx)):
        dirs = [0]*6
        adjacent_blks = [None]*6
        cur_blk_idx = total_idx[i]
    
        for d in range(3):
            if (cur_blk_idx[d] > 0) and (cur_blk_idx[d] <= block_dims[d]-1):
                dirs[d*2] = 1
                adj_blk_idx = cur_blk_idx.copy()
                adj_blk_idx[d] -= 1
                adjacent_blks[d*2] = adj_blk_idx
                all_edges.append([cur_blk_idx, adj_blk_idx, d*2])
                
                
            if cur_blk_idx[d] < block_dims[d]-1:
                dirs[(d*2)+1] = 1
                adj_blk_idx = cur_blk_idx.copy()
                adj_blk_idx[d] += 1
                adjacent_blks[(d*2)+1] = adj_blk_idx
                all_edges.append([cur_blk_idx, adj_blk_idx, (d*2)+1])
    
        all_dirs.append(dirs)
        all_adj_blks.append(adjacent_blks)
    
    return all_dirs, all_adj_blks, all_edges



def remove_reverse_edges(edges):
    
    new_edges = []  ## edges only directed in forward directions
    edge_remove = []
    for i, e in enumerate(edges):
        ## scan for adjacent edges
        for j, f in enumerate(edges):
            if not f is e:
                if np.array_equal(f[0], e[1]) and np.array_equal(f[1], e[0]):
                    edge_remove.append((i,j))

    
    for i, er in enumerate(edge_remove):
        if er[0] < er[1]:
            new_edges.append(edges[er[0]])

    return new_edges


    
def find_cull_directions(i, total_idx, block_dims):
    cur_blk_idx = total_idx[i]
    dirs = [0]*6
    
    for d in range(3):
        if (cur_blk_idx[d] > 0) and (cur_blk_idx[d] <= block_dims[d]-1):
            dirs[d*2] = 1          
            
        if cur_blk_idx[d] < block_dims[d]-1:
            dirs[(d*2)+1] = 1
            
    return np.array(dirs)


def build_connected_blocks(cur_edge, blocks_bbox, block_dims, total_idx, dir_dict):
    
    adj = (-1, 0, 1)
    
    cur_adj_blks = []  ## list of connected blocks
    cur_adj_blks_id = []  ## idx of connected blocks

    edge_dir = cur_edge[2]
    p, q = dir_dict[edge_dir]
    
    ## check if the current block even exists:
    
    blk_id = np.where((total_idx == cur_edge[0]).all(axis=1))[0][0]
    if len(blocks_bbox[blk_id]) > 0:
        cur_adj_blks.append(cur_edge[0])
        cur_adj_blks_id.append(blk_id)
        for a in adj:
            for b in adj:
                adj_edge = cur_edge[1].copy()
                adj_edge[p] += a
                adj_edge[q] += b
                if (adj_edge[p] >= 0) and (adj_edge[p] <= block_dims[p]-1):
                    if (adj_edge[q] >= 0) and (adj_edge[q] <= block_dims[q]-1):
                        ## check if the block exists
                        adj_blk_id = np.where((total_idx == adj_edge).all(axis=1))[0][0]
                        if len(blocks_bbox[adj_blk_id]) > 0:
                            cur_adj_blks.append(adj_edge)
                            cur_adj_blks_id.append(adj_blk_id)
    else:
        return [], [], []

    return edge_dir, cur_adj_blks, cur_adj_blks_id


def calculate_overlapping_regions(origin_block, cur_coords, edge_dir, overlap, dict_oc):
    oc = []
    
    for d in range(3):
        if edge_dir == dict_oc[d]:
            oc.append([cur_coords[d,1]-overlap[d]*2, cur_coords[d,1]])
        else:
            oc.append([cur_coords[d,0], cur_coords[d,1]])
            
    return np.array(oc) ## bounds of the overlapping region



