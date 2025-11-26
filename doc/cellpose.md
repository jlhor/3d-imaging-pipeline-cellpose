## Cellpose Guide

### Installation

> [!IMPORTANT]
> We strongly recommend that a new Conda environment is created (separate from the original Stardist version) for the purpose of Cellpose workflow. The required dependencies are different and some of the shared packages may be incompatible. The versions listed in this guide represent the working pipeline that we have tested.

A Conda virtual environment needs to be set up on the HPC cluster. We recommend following the instructions in the [Cellpose repository](https://github.com/MouseLand/cellpose) *and* the additional steps in the *GPU version (CUDA) on Windows or Linux* section to ensure the proper installation of GPU-enabled Cellpose package.

> [!IMPORTANT]
> For GPU support to be enabled for Pytorch during Cellpose installation, the steps above must be performed on a GPU node on the HPC. We recommend launching an interactive job on a GPU node on your HPC cluster and proceed with the installation from there.



### Additional Python dependencies

After Cellpose has been successfully installed on the Conda environment (accessible with the `conda activate <env_name>` command), the following Python packages can be installed using the `pip install` command:

```
numpy (>=1.22.1)
scipy (>=1.10.1)
scikit-image (>= 0.21.0)
numba (>=0.58.1)
pandas (>=2.0.3)
tables (>=3.8.0)
h5py (>=2.10.0)
tifffile (>=2023.7.10)
zarr (>=2.13.1)
dask (>=2023.5.0)
dask-jobqueue (>=0.8.2)
distributed (>=2023.5.0)
pyyaml (>=6.0.1)
ruamel-yaml (>=0.18.6)
tqdm (>=4.66.2)
```

### Cloning the `3d-imaging-pipeline-cellpose` repository

Navigate to the desired directory and execute the command `git clone https://github.com/jlhor/3d-imaging-pipeline-cellpose.git` to clone the entire repository to the current directory.

If the Conda environment has not been activated, use the command `conda activate <env_name>` to activate the virtual environment containing the installed packages. The Python scripts for `3d-imaging-pipeline-cellpose` are now ready to be executed.


## Segmentation with Cellpose

Cellpose segmentation can be performed via a pre-trained model (e.g. `cyto3`), or by loading a custom model trained using [Cellpose](https://github.com/MouseLand/cellpose). Model selection is specified in the configuration file.

### Cellpose Segmentation workflow: Overview

This workflow first segments the image in overlapping 3D blocks in a distributed manner

The Cellpose segmentation module is executed in two steps:
1. Overlapping 3D blocks are first segmented in a distributed manner. The segmentation labels are saved in separate 3D blocks/tiles that are individually retrieved during the data extraction step.
2. Segmented cells that overlap between blocks are subsequently removed to ensure that only one copy of label exists among all overlapping 3D blocks.

### Segmentation tutorial (using a pre-trained model)

1. Copy the image file containing the segmentation channel to `3d-imaging-pipeline-cellpose/Projects/<project_name>/input`
   (Currently, only the `.ims` format is supported)
2. Configure the `segmentation_<project_name>.yaml` file with a text editor.   
   - Specify the `InputImage` name contained within the `input` sub-directory, and the `InputChannel` with an integer corresponding to the channel order within the image (first channel is `0`, and so on). If the image contains only one channel (e.g. `.tif` file), use `0`.
   - Under `CELLPOSE_MODEL`, set `ModelType` to `library` to utilize pre-trained model, which is then specified with `ModelName` (i.e. `cyto3` in this case). `ModelDir` is ignored when `library` (pre-trained model) is selected.
   - Configure the HPC resources under the `DASK` section based on the specific cluster and infrastructure available. Note that `gpu_type` and `cpu_type` vary between institutions, and should be modified as such to request for the exact type of nodes available.   

3. Navigate to the `3d-imaging-pipeline-cellpose` root folder containing the `run_cellpose_segmentation.py` Python script and the `script_cellpose_segmentation_<project_name>.sh` shell script.

4. Execute the command `sbatch script_cellpose_segmentation_<project_name>.sh` to start a Slurm job for the segmentation module.
   - The log and error files will be output as `seg_<project_name>.out` and `seg_<project_name>.err` respectively.
5. Upon completion of the script, the output `<project_name>_prediction.zarr` file containing the segmented labels of the single cells will be generated in the `output` sub-directory for downstream processing and analysis.

## Single cell data extraction

The data extraction module follows the segmentation module by accessing the segmented labels to extract individual image channel information for single cells. This requires the `<project_name>_prediction.zarr` label file generated in the previous segmentation step.

### Data extraction: Overview

Similar to the StarDist3D version, individual cell patch (a small 3D block) that extends beyond the boundaries of 3D cell masks will be extracted from individual image channels based on the cell mask positions. In the Cellpose version, however, the segmentation labels are stored as separate image sub-blocks, and the marker intensity extraction for all cells are performed in a block-wise manner (and parallelized with HPC), and the cell data from all image sub-blocks are combined into a single output array.

When using Cellpose to segment membrane markers, the segmented labels should encompass the shape of the cells, and in this case (unlike nuclear segmentation), additional masks are generated through morphological erosion to estimate the nucleus of the cell, and subtraction of the generated masks also creates a membrane/cytoplasmic masks.

Four layers of cell masks are generated:   
   
   | Mask | Description |
   | -- | -- |
   | `segmented` | the original masks predicted by the segmentation step |
   | `cell` | slight dilation of the original `segmented` masks to encompass the membrane/cytoplasmic region of the cells |
   | `nuclear` | slight erosion of the original `segmented` masks. |
   | `membrane` | membrane/cytoplasmic mask resulting from a subtraction of the  `nuclear` mask from the dilated `cell` mask. Provides a more accurate quantification of the membrane protein staining |   
   
The output array will comprise the mean intensity value of each channel for the four separate masks for each cell.
   - This is calculated by dividing the sum of all masked voxel intensities by the sum of all mask voxels

The coordinates for each cell in `(x, y, z)` are also calculated and exported as the coordinate parameters. Conversion between `image` coordinates and `world` coordinates (in µm) can be specified in the configuration file.

### Data extraction tutorial

1. If the input image file is not the same as the input image used for segmentation, copy the image file(s) where the channels will be used for data extraction into the `input` sub-directory.
   - Note that more than one images can be used for data extraction.

> [!NOTE]
> Currently only `.ims` format is supported in this module. Support for OME-TIFF format is  under development. We recommend converting image datasets into `.ims` format using the free [Imaris File Converter](https://imaris.oxinst.com/microscopy-imaging-software-free-trial#file-converter) tool if needed.

2. Configure the `extraction_<project_name>.yaml` file contained within the `Projects/<project_name>` directory.
   - Provide the `InputImage` as a list of image file names.
   - `Channels` specify the channels to be extracted (first channel is `0`).   
     If more than one input images are used, provide the channel numbers in multiple sub-lists. Default is `'all'` to extract all channels.
   - `ChannelNames` can be provided as a list of strings indicating the channel names, or use `auto` to automatically extract the channel names from the `.ims` metadata
   - `PredictionFileName` should point to the `prediction.zarr` label file generated from the segmentation step in the `output` sub-directory. No change is needed if following directly from the previous segmentation step.
   - `OutputFilePrefix` will specify the prefix of the **final output array** file name.
   - `CellCoordinates` can be set to `world` if the positions in µm distance is desired.
   - `VoxelDimensions` can be provided as a list in `[x, y, z]` dimensions for conversion into `world` coordinates. Set to `auto` to automatically retrieve from the `.ims` metadata.
   - `OutputCSV` can be set to `True` if export to individual `.csv` dataframes is desired.
   - Configure the HPC resources under the `DASK` section based on the specific cluster and infrastructure available. Note that `gpu_type` and `cpu_type` vary between institutions, and should be modified as such to request for the exact type of nodes available.   
  
3. Navigate to the `3d-imaging-pipeline-cellpose` root folder containing the `run_cellpose_extraction.py` Python script and the `script_extraction_<project_name>.sh` shell script.
5. Execute the command `sbatch script_extraction_<project_name>.sh` to start a Slurm job for the data extraction module.
   - The log and error files will be output as `ext_<project_name>.out` and `ext_<project_name>.err` respectively.
6. Upon completion of the script, the output `<output_prefix>.h5` file containing the extracted information of the single cells will be generated in the `output` sub-directory for subsequent analysis.

### Output array

The output array will be exported in a HDF5 container (`.h5`) with multiple 2D `pandas` dataframe that can be retrived with the appropriate keys using the following command: `pandas.read_hdf('output_array.h5', key='nuclear')`

The keys for the datasets are as follows:
| Key | Description |
| -- | -- |
| `positions` |  X, Y, Z coordinates for each cell (Conversion from image coordinates to world coordinates can be set in the configuration file) |
| `indices` |  The block number and cell number (label value) of the segmented mask in each image sub-block |
| `segmented` | Mean intensity extracted from each channel based on the initial segmentation mask of the cell |
| `membrane` | Mean intensity extracted from each channel based on the membrane/cytoplasmic mask of the cell |
| `cell` | Mean intensity extracted from each channel based on the segmented + membrane/cytoplasmic mask of the cell |
| `nuclear` | Mean intensity extracted from each channel based on the eroded segmented mask of the cell |

To export the cell coordinates as world coordinates (in µm scale), set `CellCoordinates` to `world` and `VoxelDimensions` can be defined as an `[X, Y, Z]` list, or set to `auto` to automatically retrieve the voxel dimensions from the image file (Imaris only).   

To export the output arrays as `.csv` format, set `OutputCSV` to `True` in the configuration file.
