import os
import gc
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from netCDF4 import Dataset as NCDataset
from tqdm import tqdm
from copy import deepcopy
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from piv_dataset import PIVInference
from scipy.io import loadmat


# Local imports
from goflow_core import (
    dx_kernel, dy_kernel,
    compute_velocity_gradients, compute_derived_fields,
    create_tukey_window, create_boundary_mask,
    gradient_loss, compute_gradient_r2, to_numpy,
    load_datasets, create_dataloaders,
    initialize_model, save_model, load_model,
    get_model_string, count_parameters, make_splits, setup_device
)
from spectral_loss import spectral_loss
from utils import cosineSGDR
from dataSST import SatelliteDataset, writeGridSat
from writenc import ncCreate, addVal

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train GOFLOW model')
    
    # Device and model selection
    parser.add_argument('--cuda', type=int, default=0, help='CUDA device index')
    parser.add_argument('--model', type=str, default='unet',
                        choices=['unet', 'samudra0', 'samudraR', '2layer'],
                        help='Model architecture')
    parser.add_argument('--model_file', type=str, help='.pth model weight file')
    parser.add_argument('--nbase', type=int, default=16, help='Base channels for UNet')
    parser.add_argument('--kernel_size', type=int, default=5, help='Kernel size for CNN')
    
    # Loss configuration
    # parser.add_argument('--c_spec', type=float, default=0.0,
    #                     help='Spectral/gradient loss weight (0-1)')
    # parser.add_argument('--use_grad_loss', action='store_true',
    #                     help='Use gradient loss instead of spectral loss')
    
    # Training parameters
    # parser.add_argument('--epochs', type=int, default=None,
    #                     help='Number of epochs (default: 100 for c_spec=0, 50 otherwise)')
    # parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate')
    parser.add_argument('--batch_size', type=int, default=None, help='Training batch size')
    # parser.add_argument('--tcycle', type=int, default=5, help='Cosine annealing cycle length')
    # parser.add_argument('--resume', action='store_true', help='Set to resume training from previous best model' )
    # parser.add_argument('--resume_from_idx', type=int, default = None, 
    #                     help='Resume training from the model with the specified index in the output directory')
    # parser.add_argument('--resume_from_file', type=str, default = None,
    #                     help='Resume training from the model at the specified path')
    # parser.add_argument('--eval_criterion', type=str, default = 'r2',
    #                     choices=['r2','mean'], 
    #                     help='Criterion used to determin if the current model is the "best" and should be saved')
    
    # Data paths
    # parser.add_argument('--llc_file', type=str, default='llcGoes_gradT_trunc.nc',
    #                     help='LLC training data file')
    parser.add_argument('--data_root', type=str, default=None,
                        help='training dataset root directory')
    parser.add_argument('--output_dir', type=str, default='./out/output/',
                        help='Output directory where model files will be stored and the log file will be stored.')
    # parser.add_argument('--write_log', action='store_true', help = 'Write a summary of results to a log file in the directory specified by --output_dir')
    
    # Data parameters
    parser.add_argument('--nframes', type=int, default=2, help='Number of input frames')
    parser.add_argument('--step0', type=int, default=1, help='Time step stride')
    parser.add_argument('--pm', type=float, default=1, help='X grid metric')
    parser.add_argument('--pn', type=float, default=1, help='Y grid metric')

    parser.add_argument("--subsets", nargs="+", default=[], help="Subdirectories to use")
    parser.add_argument("--ext", type=str, default="tif", help="Image extension")
    parser.add_argument("--crop_size", type=int, nargs=2, default=[256, 256])
    parser.add_argument("--rand_trans", nargs='+', type=int, default=0)
    parser.add_argument('--mat_val', action='store_true',
                        help='compare inferred field to manually selected particle loctions in .mat file')

    
    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_args()

    # set up gpu device
    device = setup_device(args.cuda)

    dataset = PIVInference(args.data_root, subsets=args.subsets, ext=args.ext)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=5)


    n_input = 2
    n_output = 2
    model_prev = initialize_model(
        n_input, n_output,
        model_name=args.model,
        nbase=args.nbase,
        kernel_size=args.kernel_size,
        device=device
    )
    # model = model_prev
    model = torch.compile(model_prev)


    model = load_model(model, args.model_file, device)

    out_list = []
    name_list = []

    model.eval()
    with torch.no_grad():
        for (imgs, name) in tqdm(loader, desc = 'PIV inference'):
            imgs = imgs.to(device)
            out = model(imgs)
            out_list.append(out.cpu().numpy())
            name_list.append(name)
    
    out = np.concatenate(out_list, axis = 0)
    name = np.concatenate(name_list, axis=0)    

    dir = Path(args.output_dir)
    
    for (o,n) in zip(out,name):
        u = o[0,:,:]
        v = o[1,:,:]

        umax = np.max(u)
        vmax = np.max(v)
        umin = np.min(u)
        vmin = np.min(v)

        plt.figure(1)
        im = plt.imshow(u, vmin = umin, vmax = umax)
        cbar = plt.colorbar(im)
        cbar.set_label('u', rotation=270, labelpad=15)

        plt.figure(2)
        im = plt.imshow(v, vmin = vmin, vmax = vmax)
        cbar = plt.colorbar(im)
        cbar.set_label('v', rotation=270, labelpad=15)


        if args.mat_val:
            mat = loadmat(os.path.join(dir / f"{n}.mat"))
            p = np.array(mat['p'])
            uv = p[:,1,:]-p[:,0,:]
            NP = mat['np']

            for i in range(0, NP[0,0]-1):
                ui = uv[i,0]
                vi = uv[i,1]
                # print(f"u error {u[p[i,0,1],p[i,0,0]]-uv[i,0,0]}")
                plt.figure(1)
                plt.scatter(p[i,0,0]-50,p[i,0,1]-35,c=ui-10, s=10, vmin = umin, vmax = umax)

                plt.figure(2)
                plt.scatter(p[i,0,0]-53,p[i,0,1]-35,c=vi, s=10, vmin = vmin, vmax = vmax)



                
        
        plt.figure(1)
        plt.savefig(os.path.join(dir / f"{n}_u.png"))
        plt.close()
        
        plt.figure(2)
        plt.savefig(os.path.join(dir / f"{n}_v.png"))
        plt.close()
                




if __name__ == '__main__':
    main()



