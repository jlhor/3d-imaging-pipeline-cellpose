# -*- coding: utf-8 -*-

import numpy as np
from tqdm import tqdm

from cellpose_segmentation.overlap_utils import find_adjacent_blocks, remove_reverse_edges, build_connected_blocks, calculate_overlapping_regions, _label_overlap


def get_overlap_stack(blk, zarr_seg, oc, img_shape):
    

    tmp_coords = np.array(zarr_seg[f'block_{blk}']['blk_coords'])
    
    ## make sure the max coords are not larger than image shape
    for d in range(3):
        if tmp_coords[d,1] > img_shape[d]:
            tmp_coords[d,1] = img_shape[d]
    
    crop_start = []
    overlap_start = []
    
    crop_end = []
    overlap_end = []
    
    crop_slice = []
    overlap_slice = []
    
    
    for d in range(3):
        if tmp_coords[d,0] > oc[d,0] :
            crop_start.append(0)
            overlap_start.append(tmp_coords[d,0] - oc[d,0])
        else:
            crop_start.append(oc[d,0] - tmp_coords[d,0])
            overlap_start.append(0)
            
        if tmp_coords[d,1] > oc[d,1]:
            crop_end.append(oc[d,1] - tmp_coords[d,0])
            overlap_end.append(oc[d,1] - oc[d,0])
        else:
            crop_end.append(tmp_coords[d,1] - tmp_coords[d,0])
            overlap_end.append(tmp_coords[d,1] - oc[d,0])
    
    for d in range(3):
        crop_slice.append(slice(crop_start[d], crop_end[d]))
        overlap_slice.append(slice(overlap_start[d], overlap_end[d]))
    
    crop_slice = np.s_[tuple(crop_slice)]
    overlap_slice = np.s_[tuple(overlap_slice)]
  
    tmp_masks = np.array(zarr_seg[f'block_{blk}']['mask'])
    
    oc_shape = tuple(oc[:,1] - oc[:,0])
    tmp_overlap = np.zeros((oc_shape), dtype=tmp_masks.dtype)
    
    tmp_overlap[overlap_slice] = tmp_masks[crop_slice]
    
    return tmp_overlap


def calculate_overlap(b, blk, oc_labels, overlap_labels):
    oc_a = oc_labels[0].copy()
    oc_b = oc_labels[b].copy()
    
    overlap_ab = _label_overlap(oc_a, oc_b)

    n_pixels_b = np.sum(overlap_ab, axis=0, keepdims=True)
    n_pixels_a = np.sum(overlap_ab, axis=1, keepdims=True)
    
    iou = overlap_ab / (n_pixels_a + n_pixels_b - overlap_ab)
    iou[np.isnan(iou)] = 0.0

    iou_filtered = np.where(iou > 0.1)
    
    oc_a_unique = np.unique(iou_filtered[0])
    oc_b_matches = []
    for a in oc_a_unique:
        mask_a = iou_filtered[0]==a
        tmp_match = iou_filtered[1][mask_a]
        tmp_sizes = n_pixels_b[0, iou_filtered[1][mask_a]]
        oc_b_matches.append([tmp_match, tmp_sizes])
    
    b_max_len = np.max([ len(x) for x, y in oc_b_matches ])
    b_matches = np.zeros((len(overlap_labels), b_max_len), dtype=np.int32)
    b_sizes = np.zeros((len(overlap_labels), b_max_len), dtype=np.int32)
    b_blk_id = np.zeros((len(overlap_labels), b_max_len), dtype=np.int32)
    b_blk_id[...] = -1
    
    for a, aa in enumerate(oc_a_unique):
        b_matches[aa,:len(oc_b_matches[a][0])] = oc_b_matches[a][0]
        b_sizes[aa,:len(oc_b_matches[a][0])] = oc_b_matches[a][1]
        b_blk_id[aa,:len(oc_b_matches[a][0])] = blk
        

    return b_matches, b_sizes, b_blk_id, n_pixels_a

def remove_overlaps(total_idx, block_dims, overlap, img_shape, zarr_seg):
    
    dirs, adj_blks, edges = find_adjacent_blocks(total_idx, block_dims)
    
    ## remove duplicates/reverse direction edges
    
    new_edges = remove_reverse_edges(edges)
    
    ## pre-load bbox and labels from zarr into memory
    
    print('Preloading from zarr into memory')
    
    blocks_bbox = [[]] * len(total_idx)
    blocks_labels = [[]] * len(total_idx)
    
    for i in range(len(total_idx)):
        if 'labels' in zarr_seg[f'block_{i}']:
            blocks_bbox[i] = np.array(zarr_seg[f'block_{i}']['bbox'])
            blocks_labels[i] = np.array(zarr_seg[f'block_{i}']['labels'])


    for ed in tqdm(range(len(new_edges)), desc='Edge operation'):
        #print(ed)
        cur_edge = new_edges[ed]

        ## edge direction determines the other two axes (9 potential overlapping adjacent blocks)
        
        e_dir = { 1 : (1,2),
                  3 : (0,2),
                  5 : (0,1)}
        
        ## build list of connected blocks
        
        edge_dir, cur_adj_blks, cur_adj_blks_id = build_connected_blocks(cur_edge,
                                                                         blocks_bbox,
                                                                         block_dims,
                                                                         total_idx,
                                                                         dir_dict=e_dir)
        
        
        ## check if the block exists
        if len(cur_adj_blks_id) > 0:
            ## calculate overlapping region
            
            ## we always start from the origin block (cur_adj_blks_id[0]) and move in forward directions from there
            
            oc_dir = { 0: 1,
                       1: 3,
                       2: 5}
            
            cur_coords = np.array(zarr_seg[f'block_{cur_adj_blks_id[0]}']['blk_coords'])
            
            oc = calculate_overlapping_regions(cur_adj_blks_id[0],
                                               cur_coords,
                                               edge_dir=edge_dir,
                                               overlap=overlap,
                                               dict_oc=oc_dir)
      

            ##
            
            oc_labels = []
            init_labels = None
            
            for b, blk in enumerate(cur_adj_blks_id):
                
                tmp_overlap = get_overlap_stack(blk, zarr_seg, oc, img_shape)
                oc_labels.append(tmp_overlap)
                if b == 0:
                    init_labels = np.unique(tmp_overlap)

            oc_labels = np.stack(oc_labels)

        #############################

            overlap_labels = np.zeros((np.max(init_labels)+1, 1), dtype=int)
            overlap_labels[init_labels,0] = init_labels
            overlap_blk_id = np.zeros((np.max(init_labels)+1, 1), dtype=int)
            overlap_blk_id[init_labels,0] = cur_adj_blks_id[0]
            overlap_sizes = np.zeros((np.max(init_labels)+1, 1), dtype=int)

            for b, blk in enumerate(cur_adj_blks_id):
                if b > 0:
                    b_matches, b_sizes, b_blk_id, init_pixels = calculate_overlap(b, blk, oc_labels, overlap_labels)
                        
                    if b == 1:
                        overlap_sizes[:,0] =  init_pixels.ravel()
                        
                    overlap_labels = np.concatenate((overlap_labels, b_matches), axis=1)
                    overlap_sizes = np.concatenate((overlap_sizes, b_sizes), axis=1)
                    overlap_blk_id = np.concatenate((overlap_blk_id, b_blk_id), axis=1)
                    
            #### check if the cells still exist / have been erased
    
            for b, blk in enumerate(cur_adj_blks_id):
                tmp_labels = blocks_labels[blk]
                blk_mask = overlap_blk_id == blk
                label_excluded = [ x for x in overlap_labels[blk_mask] if not x in tmp_labels and not x == 0]
                tmp_lbl_mask = np.isin(overlap_labels[blk_mask], np.array(label_excluded))
                
                cbl = overlap_labels[blk_mask]
                cbl[tmp_lbl_mask] = 0
                overlap_labels[blk_mask] = cbl
                
                cbs = overlap_sizes[blk_mask]
                cbs[tmp_lbl_mask] = 0
                overlap_sizes[blk_mask] = cbs

    
            ## collapse array by removing blank entries
            selected_rows = np.sum(overlap_labels, axis=1) > 0
    
            overlap_labels = overlap_labels[selected_rows]
            overlap_sizes = overlap_sizes[selected_rows]
            overlap_blk_id = overlap_blk_id[selected_rows]
            ### find the maximum size cells and add the excluded cells into cull list
    
            to_cull = []
    
            for r in range(len(overlap_labels)):
                nz = np.nonzero(overlap_labels[r])[0]
                ms = np.argmax(overlap_sizes[r])
                for n in nz:
                    if n != ms:
                        to_cull.append((overlap_blk_id[r,n], overlap_labels[r,n]))
            
            if len(to_cull) > 0:
                to_cull = np.array(to_cull)
        
                ## update block labels
                for b, blk in enumerate(cur_adj_blks_id):
                    to_cull_blk = to_cull[to_cull[:,0] == blk]
                    
                    new_assigned = [ (x, y) for x, y in zip(blocks_labels[blk], blocks_bbox[blk]) if not x in to_cull_blk[:,1] ]
                    
                    if len(new_assigned) > 0:
                        new_label, new_bbox = zip(*new_assigned)
                        new_label = np.array(new_label)
                        new_bbox = np.array(new_bbox)
                        
                        blocks_labels[blk] = new_label[...]
                        blocks_bbox[blk] = new_bbox[...]
                
        
    return blocks_bbox, blocks_labels