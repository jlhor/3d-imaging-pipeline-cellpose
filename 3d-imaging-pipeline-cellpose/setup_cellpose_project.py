# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 04:37:36 2024

@author: horj2
"""


import sys
from pathlib import Path
from ruamel.yaml import YAML
import os

root_path = Path(os.getcwd())

config_path = root_path / Path('config')


def run(project_name):
    
    ## check if main project path exists
    main_project_path = root_path / Path('Projects')
    main_project_path.mkdir(parents=True, exist_ok=True)
    
    print(f'Generating cd folders for project {project_name}')
    project_dir = main_project_path / project_name
    project_dir.mkdir()
    
    sub_dirs = [ project_dir / Path('input'),
                 project_dir / Path('output'),
                 project_dir / Path('models'),
                 project_dir / Path('temp') ]
    
    for d in sub_dirs:
        d.mkdir()
    
    print('Generating configuration files')
    out_seg_cfg = project_dir / Path(f'segmentation_{project_name}.yaml')
    out_ext_cfg = project_dir / Path(f'extraction_{project_name}.yaml')
    
    yaml_obj = YAML()
    yaml_obj.default_flow_style = False
    yaml_obj.preserve_quotes = True
    yaml_obj.indent(mapping=4)
    seg_cfg = yaml_obj.load(config_path / Path("hpc_cellpose_prediction_template.yaml"))
    
    seg_cfg['ProjectName'] = project_name
    seg_cfg['ProjectPath'] = str(project_dir)
    
    seg_cfg['PredictionFileName'] = f'{project_name}_prediction.zarr'
    
    yaml_obj.dump(seg_cfg, out_seg_cfg)

    ext_cfg = yaml_obj.load(config_path / Path("hpc_cellpose_extraction_template.yaml"))
    ext_cfg['ProjectName'] = project_name
    ext_cfg['ProjectPath'] = str(project_dir)
    ext_cfg['PredictionFileName'] = f'{project_name}_prediction.zarr'
    ext_cfg['ExtractionFileName'] = f'{project_name}_extraction.zarr'
    ext_cfg['OutputFilePrefix'] = f'{project_name}_output_array'
    
    yaml_obj.dump(ext_cfg, out_ext_cfg)
                  
    print('Generating scripts')
 
    sh_seg_path = root_path / Path(f'script_segmentation_{project_name}.sh')
    sh_ext_path = root_path / Path(f'script_extraction_{project_name}.sh')
    
    with open(sh_seg_path, 'w') as rsh:
        rsh.write(
f'''#!/bin/bash
#SBATCH --cpus-per-task=56
#SBATCH --mem=200g
#SBATCH --constraint='x2695'
#SBATCH --time=2-0:00:00
#SBATCH -o seg_{project_name}.out
#SBATCH -e seg_{project_name}.err

source /data/{os.environ.get('USER')}/miniconda3/etc/profile.d/conda.sh
conda activate {os.environ['CONDA_DEFAULT_ENV']}
cd {root_path}
export OMP_NUM_THREADS=14
python -m run_cellpose_segmentation "{out_seg_cfg.relative_to(root_path)}"
''')
    
    with open(sh_ext_path, 'w') as rsh:
            rsh.write(
f'''#!/bin/bash
#SBATCH --cpus-per-task=56
#SBATCH --mem=120g
#SBATCH --constraint='x2695'
#SBATCH --time=2-0:00:00
#SBATCH -o ext_{project_name}.out
#SBATCH -e ext_{project_name}.err

source /data/{os.environ.get('USER')}/miniconda3/etc/profile.d/conda.sh
conda activate {os.environ['CONDA_DEFAULT_ENV']}
cd {root_path}
python -m run_cellpose_extraction "{out_ext_cfg.relative_to(root_path)}"
''')
    
    print('Done.')

if __name__ == '__main__':
    run(sys.argv[1])
    
