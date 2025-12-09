# -*- coding: utf-8 -*-

from dask.distributed import Client, LocalCluster, as_completed
from dask_jobqueue import SLURMCluster

##############

def create_gpu_cluster(mode, config=None):
    if mode == 'local':
        cluster = LocalCluster(n_workers=1,
                       threads_per_worker=None,
                       memory_limit=None,
                       processes=False,)

    elif mode == 'SLURM':
        if config:
            cores = config['cores']
            processes = config['processes']
            memory_limit = config['memory']
            walltime = config['walltime']    
            gpu_type = config['gpu_type']
        else:
            processes = 1
            cores = 28
            memory_limit = "120GB"
            walltime = "24:00:00"
            gpu_type = "k80"
            
        cluster = SLURMCluster(
                       cores=cores,
                       memory=memory_limit,
                       processes=processes,
                       walltime=walltime,
                       queue="gpu",
                       job_extra_directives=[f'--gres=gpu:{gpu_type}:1'])
        
    
    return cluster


def create_cluster(mode, config=None):
    if mode == 'SLURM':   
        if config:
            cores = config['cores']
            processes = config['processes']
            memory_limit = config['memory']
            walltime = config['walltime']
            if 'cpu_type' in config.keys():
                cpu_type = config['cpu_type']
            else:
                cpu_type = "x2695"
        else:
            processes = 1
            cores = 56
            memory_limit = "240GB"
            walltime = "24:00:00"
            cpu_type = "x2695"

        cluster = SLURMCluster(
                       cores=cores,
                       memory=memory_limit,
                       processes=processes,
                       walltime=walltime,
                       job_extra_directives=[f"--constraint='{cpu_type}'"])

    
    return cluster

