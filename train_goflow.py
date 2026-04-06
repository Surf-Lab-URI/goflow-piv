"""
GOFLOW Training Script
======================
Training pipeline for GOFLOW (Geostationary Ocean Flow) models.

Usage:
    python train_goflow.py --cuda 0 --model unet --c_spec 0.5 --nbase 16

Author: Kaushik (UCLA Atmospheric and Oceanic Sciences)
"""

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


# Local imports
from goflow_core import (
    dx_kernel, dy_kernel,
    compute_velocity_gradients, compute_derived_fields,
    create_tukey_window, create_boundary_mask,
    gradient_loss, compute_gradient_r2, to_numpy,
    load_datasets, create_dataloaders,
    initialize_model, save_model, load_model,
    get_model_string, count_parameters, make_splits
)
from spectral_loss import spectral_loss
from utils import cosineSGDR
from dataSST import SatelliteDataset, writeGridSat
from writenc import ncCreate, addVal


# =============================================================================
# Configuration
# =============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train GOFLOW model')
    
    # Device and model selection
    parser.add_argument('--cuda', type=int, default=0, help='CUDA device index')
    parser.add_argument('--model', type=str, default='unet',
                        choices=['unet', 'samudra0', 'samudraR', '2layer'],
                        help='Model architecture')
    parser.add_argument('--nbase', type=int, default=16, help='Base channels for UNet')
    parser.add_argument('--kernel_size', type=int, default=5, help='Kernel size for 2layer CNN')
    
    # Loss configuration
    parser.add_argument('--c_spec', type=float, default=0.0,
                        help='Spectral/gradient loss weight (0-1)')
    parser.add_argument('--use_grad_loss', action='store_true',
                        help='Use gradient loss instead of spectral loss')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of epochs (default: 100 for c_spec=0, 50 otherwise)')
    parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate')
    parser.add_argument('--batch_size', type=int, default=None, help='Training batch size')
    parser.add_argument('--tcycle', type=int, default=5, help='Cosine annealing cycle length')
    parser.add_argument('--resume', action='store_true', help='Set to resume training from previous best model' )
    parser.add_argument('--resume_from_idx', type=int, default = None, 
                        help='Resume training from the model with the specified index in the output directory')
    parser.add_argument('--resume_from_file', type=str, default = None,
                        help='Resume training from the model at the specified path')
    parser.add_argument('--eval_criterion', type=str, default = 'r2',
                        choices=['r2','mean'], 
                        help='Criterion used to determin if the current model is the "best" and should be saved')
    
    # Data paths
    # parser.add_argument('--llc_file', type=str, default='llcGoes_gradT_trunc.nc',
    #                     help='LLC training data file')
    parser.add_argument('--data_root', type=str, default=None,
                        help='training dataset root directory')
    parser.add_argument('--output_dir', type=str, default='./output/',
                        help='Output directory where model files will be stored and the log file will be stored.')
    parser.add_argument('--write_log', action='store_true', help = 'Write a summary of results to a log file in the directory specified by --output_dir')
    
    # Data parameters
    parser.add_argument('--nframes', type=int, default=3, help='Number of input frames')
    parser.add_argument('--step0', type=int, default=1, help='Time step stride')
    parser.add_argument('--pm', type=float, default=1, help='X grid metric')
    parser.add_argument('--pn', type=float, default=1, help='Y grid metric')

    parser.add_argument("--subsets", nargs="+", default=[], help="Subdirectories to use")
    parser.add_argument("--ext", type=str, default="tif", help="Image extension")
    parser.add_argument("--crop-size", type=int, nargs=2, default=[256, 256])
    parser.add_argument("--rand_trans", nargs='+', type=int, default=0)

    
    return parser.parse_args()


def setup_device(cuda_idx: int) -> torch.device:
    """Configure CUDA device and clear memory."""
    device = torch.device(f'cuda:{cuda_idx}' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(cuda_idx)
        torch.cuda.empty_cache()
    gc.collect()
    print(f'Device: {device}')
    return device


# =============================================================================
# Training Loop
# =============================================================================

def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    kernel_x: torch.Tensor,
    kernel_y: torch.Tensor,
    mask: torch.Tensor,
    tukey_window: torch.Tensor,
    c_spec: float,
    use_grad_loss: bool = False
) -> tuple[float, float]:
    """
    Run one training epoch.
    
    Returns:
        Tuple of (l1_loss, auxiliary_loss) from first batch for logging.
    """
    model.train()
    first_batch_losses = None
    # loop through batches. each iteration, load a batches worth of inputs (x),
    # targets (y), and input-specific masks (m).
    for ib, (x, y, m) in enumerate(tqdm(train_loader, desc='Training')):
        x, y, m = x.to(kernel_x.device), y.to(kernel_x.device), m.to(kernel_x.device)
        ms = torch.transpose(torch.stack((m, m), dim = 0),0,1).to(torch.float)
        m = m.to(torch.float)
        # im, rm, cm = torch.where(mask[None,:,:]*m > 0)
        y_pred = model(x)
        
        # Pointwise L1 loss with boundary masking
        loss_l1 = criterion(
            y.squeeze() * ms.squeeze() * mask[None, None, :, :],
            y_pred.squeeze() * ms.squeeze() * mask[None, None, :, :]
        )
        
        # Auxiliary loss (gradient or spectral)
        if use_grad_loss:
            loss_aux = gradient_loss(y_pred.squeeze(), y.squeeze(), criterion, 
                                     mask[None,:,:]*m, kernel_x, kernel_y)
        else:
            loss_aux = spectral_loss(y_pred*ms, y*ms, tukey_window)
        
        # Combined loss
        loss = (1 - c_spec) * loss_l1 + c_spec * loss_aux
        
        # Store first batch losses for logging
        if ib == 0:
            first_batch_losses = (loss_l1.item(), loss_aux.item())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    return first_batch_losses


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    kernel_x: torch.Tensor,
    kernel_y: torch.Tensor,
    mask: torch.Tensor,
    tukey_window: torch.Tensor
) -> tuple[float, float]:
    """
    Evaluate model on test set.
    
    Returns:
        Tuple of (mean_r2, mean_spectral_loss)
    """
    model.eval()
    total_r2 = 0.0
    total_spec_loss = 0.0
    total_mean_diff = 0.0
    count = 0
    plotcount = 0
    with torch.no_grad():
        for x, y, m in tqdm(test_loader, desc='Evaluating'):
            x, y, m = x.to(kernel_x.device), y.to(kernel_x.device), m.to(kernel_x.device)
            ms = torch.transpose(torch.stack((m, m), dim = 0),0,1).to(torch.float)
            m = m.to(torch.float)
            y_pred = model(x)
            mask_cpu = mask.to("cpu").numpy()
            
            if count % 5 == 0:
                for i in range(0,1):
                    y_cpu = y.to("cpu").numpy()[i,:,:,:]
                    x_cpu = x.to("cpu").numpy()[i,:,:,:]
                    y_pred_cpu = y_pred.to("cpu").numpy()[i,:,:,:]
                    m_cpu = m.to("cpu").numpy()[i,:,:]
                    parent_dir = Path(__file__).resolve().parent
                    dir = parent_dir / 'debugplots'
                    dir.mkdir(exist_ok=True)
                    
                    plt.figure()
                    im = plt.imshow(y_pred_cpu[0,:,:])
                    cbar = plt.colorbar(im)
                    cbar.set_label('u (m/s)', rotation=270, labelpad=15)
                    plt.title('Inference')
                    plt.savefig(dir / f"{plotcount}upred.png")
                    plt.close()

                    plt.figure()
                    im = plt.imshow(y_cpu[0,:,:])
                    cbar = plt.colorbar(im)
                    cbar.set_label('u (m/s)', rotation=270, labelpad=15)
                    plt.title('Target')
                    plt.savefig(dir / f"{plotcount}utarget.png")
                    plt.close()

                    plt.figure()
                    im = plt.imshow(y_pred_cpu[1,:,:])
                    cbar = plt.colorbar(im)
                    cbar.set_label('v (m/s)', rotation=270, labelpad=15)
                    plt.title('Inference')
                    plt.savefig(dir / f"{plotcount}vpred.png")
                    plt.close()

                    plt.figure()
                    im = plt.imshow(y_cpu[1,:,:])
                    cbar = plt.colorbar(im)
                    cbar.set_label('v (m/s)', rotation=270, labelpad=15)
                    plt.title('Target')
                    plt.savefig(dir / f"{plotcount}vtarget.png")
                    plt.close()

                    plt.figure()
                    plt.imshow(x_cpu[1,:,:], cmap='gray')
                    plt.savefig(dir / f"{plotcount}im1.png")
                    plt.close()

                    plt.figure()
                    plt.imshow(x_cpu[0,:,:], cmap='gray')
                    plt.savefig(dir / f"{plotcount}im0.png")
                    plt.close()

                    plt.figure()
                    plt.imshow(mask_cpu*m_cpu, cmap='gray')
                    plt.savefig( dir / f"{plotcount}mask.png")
                    plt.close()
                    plotcount += 1


            # Spectral loss
            spec_loss = spectral_loss(y_pred*ms, y*ms, tukey_window)
            
            # R² on gradient fields (vorticity + strain)
            r2 = compute_gradient_r2(y, y_pred, kernel_x, kernel_y, mask[None,:,:]*m)

            # Calculate mean error in the mean
            d = y_pred*ms-y*ms
            d = torch.mean(d,[2,3])
            d = torch.sqrt(d[:,0]**2 + d[:,1]**2)
            mean_diff = to_numpy(torch.mean(d))[0]
            
            total_mean_diff +=mean_diff
            total_r2 += r2
            total_spec_loss += spec_loss.item()
            count += 1
    
    return total_r2 / count, total_spec_loss / count, total_mean_diff / count


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    config: argparse.Namespace,
    device: torch.device
) -> tuple[nn.Module, np.ndarray, np.ndarray]:
    """
    Full training loop with checkpointing and evaluation.
    
    Returns:
        Tuple of (best_model, r2_history)
    """
    # Setup derivative kernels
    kernel_x = dx_kernel(config.pm).to(device)
    kernel_y = dy_kernel(config.pn).to(device)
    
    # Will be initialized on first batch
    mask = None
    tukey_window = None
    
    # Get model string for filenames
    model_str = get_model_string(config.model, config.nbase, config.kernel_size, config.use_grad_loss)
    
    # Tracking
    best_r2 = -1000
    best_spec = 1000
    best_mean_diff = 1e5
    r2_history = np.zeros(config.epochs)
    mean_history = np.zeros(config.epochs)
    best_model = None
    
    for epoch in range(config.epochs):
        # Learning rate scheduling
        lr = cosineSGDR(optimizer, epoch, T0=config.tcycle, eta_min=0, 
                        eta_max=config.lr, scheme='constant')
        
        # Initialize mask/window on first epoch using data shape
        if mask is None:
            sample_x, sample_y, m = next(iter(train_loader))
            shape = sample_y.shape[-2:]
            bw = 0
            if config.rand_trans:
                if isinstance(config.rand_trans, list):
                    bw = max(config.rand_trans)
                else:
                    bw = config.rand_trans
            mask = create_boundary_mask(shape,boundary_width=bw+2).to(device)
            tukey_window = create_tukey_window(shape).to(device)
        
        # Train one epoch
        l1_loss, aux_loss = train_epoch(
            model, train_loader, optimizer, criterion,
            kernel_x, kernel_y, mask, tukey_window,
            config.c_spec, config.use_grad_loss
        )
        loss_type = 'grad' if config.use_grad_loss else 'spec'
        print(f'Epoch {epoch+1}: L1={l1_loss:.4f}, {loss_type}={aux_loss:.4f}')
        
        # Evaluate
        r2, spec_loss, mean_diff = evaluate_model(model, test_loader, kernel_x, kernel_y, mask, tukey_window)
        r2_history[epoch] = r2
        mean_history[epoch] = mean_diff
        
        # Track best model
        # if r2 > best_r2:
        #     best_r2 = r2
        #     best_model = deepcopy(model)
            
        #     # Save checkpoint and run inference
        #     checkpoint_path = f'{model_str}_{config.step0}_{config.nframes}_{config.c_spec}cs.pth'
        #     save_model(best_model, checkpoint_path)
            
            # Write test results
            # write_test_results(
            #     epoch, best_model, test_loader, kernel_x, kernel_y,
            #     config.c_spec, model_str, config.output_dir
            # )
            
            # # Process satellite data
            # out_val, grad_val, sst_val = run_satellite_inference(
            #     best_model, config.goes_file, config.valid_inds,
            #     config.pm, config.pn
            # )
            
            # # Write satellite predictions
            # output_file = f'preds_{model_str}_{config.step0}_{config.nframes}_{config.c_spec}cs{config.goes_file}'
            # write_satellite_netcdf(output_file, out_val, grad_val, sst_val,
            #                        config.valid_inds, config.goes_file)
        
        # Track best r2, mean, spec_loss
        if epoch > 1:
            if r2 > best_r2:
                best_r2 = r2
            if spec_loss < best_spec:
                best_spec = spec_loss
            if mean_diff < best_mean_diff:
                best_mean_diff = mean_diff


        # Track best model
        if (config.eval_criterion == 'r2' and best_r2 == r2) or (config.eval_criterion == 'mean' and best_mean_diff == mean_diff):
            best_model = deepcopy(model)
            if config.write_log:
                checkpoint_path = os.path.join(config.output_subdir, f'{config.exp_idx}.pth')
                config.logdf.loc[config.exp_idx, 'best_model_file'] = checkpoint_path
                config.logdf.loc[config.exp_idx, 'mean'] = mean_diff
                config.logdf.loc[config.exp_idx, 'r2'] = r2
                config.logdf.loc[config.exp_idx, 'spec'] = spec_loss
                config.logdf.loc[config.exp_idx, 'epoch_best'] = epoch + 1
                config.logdf.to_csv(config.logpath)
                print('Updated log file with new best model')
                # print(config.logdf)
            else:
                checkpoint_path = os.path.join(config.output_subdir,f'{model_str}_{config.step0}_{config.nframes}_{config.c_spec}cs.pth')
            
            save_model(best_model, checkpoint_path)
            # Write test results
            if config.write_log:
                output_prefix = config.output_subdir
            else:
                output_prefix = os.path.join(config.output_subdir, f"test_{model_str}_{config.c_spec}cspec")
            write_test_results(
                epoch, best_model, test_loader, kernel_x, kernel_y, output_prefix, mask
            )

        config.logdf.loc[config.exp_idx,'epochs'] = epoch + 1
        config.logdf.to_csv(config.logpath)
        
        print(f'Epoch {epoch+1}/{config.epochs} | R²: {r2:.4f} (best: {best_r2:.4f}) | '
              f'Spec: {spec_loss:.4f} (best: {best_spec:.4f}) | mean difference: {mean_diff:.4f} (best:{best_mean_diff:.4f})')
    
    return best_model, r2_history, mean_history


# =============================================================================
# Satellite Data Processing
# =============================================================================

# def run_satellite_inference(
#     model: nn.Module,
#     goes_file: str,
#     valid_inds: tuple,
#     pm: float,
#     pn: float,
#     batch_size: int = 4
# ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
#     """
#     Run inference on GOES satellite data.
    
#     Returns:
#         Tuple of (velocities, gradient_fields, sst_data)
#     """
#     device = next(model.parameters()).device
#     kernel_x = dx_kernel(pm).to(device)
#     kernel_y = dy_kernel(pn).to(device)
    
#     goes_dataset = SatelliteDataset(goes_file, ['log_gradT'], valid_inds, train=False)
#     goes_loader = DataLoader(goes_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
#     out_list = []
#     grad_list = []
#     sst_list = []
    
#     model.eval()
#     with torch.no_grad():
#         for sst in tqdm(goes_loader, desc='Satellite inference'):
#             # Store input SST
#             sst_list.append(sst[:, 1, :, :].cpu().numpy()[:, None, :, :])
            
#             sst = sst.to(device)
#             out = model(sst)
#             out_list.append(out.cpu().numpy())
            
#             # Compute gradient fields
#             ux, uy, vx, vy = compute_velocity_gradients(out, kernel_x, kernel_y)
#             vort, div, strain = compute_derived_fields(ux, uy, vx, vy)
#             grad_list.append(torch.stack((vort, div, strain), dim=1).cpu().numpy())
    
#     out_val = np.concatenate(out_list, axis=0)
#     grad_val = np.concatenate(grad_list, axis=0)
#     sst_val = np.concatenate(sst_list, axis=0).squeeze()
    
#     return out_val, grad_val, sst_val


# =============================================================================
# NetCDF Output
# =============================================================================

# def write_satellite_netcdf(
#     output_file: str,
#     out_val: np.ndarray,
#     grad_val: np.ndarray,
#     sst_val: np.ndarray,
#     valid_inds: tuple,
#     goes_file: str
# ):
#     """Write satellite prediction results to NetCDF."""
#     nt, _, ny, nx = out_val.shape
#     print(f'Writing {output_file}: shape=({nt}, {ny}, {nx})')
    
#     with NCDataset(goes_file, 'r') as nch:
#         varnames = ['U', 'V', 'Vorticity', 'Divergence', 'Strain', 'BT', 'loggrad_BT']
#         nc = ncCreate(output_file, nx, ny, varnames, dt=2)
        
#         for it in tqdm(range(nt), desc='Writing NetCDF'):
#             BT = nch.variables['BT'][it + 12, 
#                                      valid_inds[0]:valid_inds[1],
#                                      valid_inds[2]:valid_inds[3]]
#             addVal(nc, 'U', out_val[it, 0, :, :], it)
#             addVal(nc, 'V', out_val[it, 1, :, :], it)
#             addVal(nc, 'Vorticity', grad_val[it, 0, :, :], it)
#             addVal(nc, 'Divergence', grad_val[it, 1, :, :], it)
#             addVal(nc, 'Strain', grad_val[it, 2, :, :], it)
#             addVal(nc, 'BT', BT, it)
#             addVal(nc, 'loggrad_BT', sst_val[it, :, :], it)
        
#         nc.close()
    
#     writeGridSat(goes_file, output_file, valid_inds)


def write_test_results(
    epoch: int,
    model: nn.Module,
    test_loader: DataLoader,
    kernel_x: torch.Tensor,
    kernel_y: torch.Tensor,
    output_prefix: str,
    mask: torch.Tensor
):
    """Write test set predictions and gradient fields to NetCDF."""
    model.eval()
    device = next(model.parameters()).device
    
    inputs_list = []
    outputs_list = []
    targets_list = []
    masks_list = []
    true_grads_list = []
    pred_grads_list = []

    #make results directory
    os.makedirs(output_prefix, exist_ok=True)

    with torch.no_grad():
        for x, y_true, m in tqdm(test_loader, desc='Processing test set'):
            x, y_true, m = x.to(device), y_true.to(device), m.to(device)
            y_pred = model(x)
            
            # Compute gradients
            ux_true, uy_true, vx_true, vy_true = compute_velocity_gradients(y_true, kernel_x, kernel_y)
            ux_pred, uy_pred, vx_pred, vy_pred = compute_velocity_gradients(y_pred, kernel_x, kernel_y)
            
            vort_true, div_true, strain_true = compute_derived_fields(ux_true, uy_true, vx_true, vy_true)
            vort_pred, div_pred, strain_pred = compute_derived_fields(ux_pred, uy_pred, vx_pred, vy_pred)
            
            inputs_list.append(x.cpu().numpy())
            outputs_list.append(y_pred.cpu().numpy())
            targets_list.append(y_true.cpu().numpy())
            masks_list.append((m*mask[None,:,:]).cpu().numpy())
            true_grads_list.append(torch.stack((vort_true, div_true, strain_true), dim=1).cpu().numpy())
            pred_grads_list.append(torch.stack((vort_pred, div_pred, strain_pred), dim=1).cpu().numpy())
        

    # Concatenate batches
    inputs = np.concatenate(inputs_list, axis=0)
    outputs = np.concatenate(outputs_list, axis=0)
    targets = np.concatenate(targets_list, axis=0)
    masks = np.concatenate(masks_list, axis=0)
    true_grads = np.concatenate(true_grads_list, axis=0)
    pred_grads = np.concatenate(pred_grads_list, axis=0)

    plotcount = 0
    num_tests_recorded = 5
    for i in range(0,inputs.shape[0],round(inputs.shape[0]/num_tests_recorded)):
        k = 0.3
        umin = np.min(targets[i,0,:,:])
        umin = umin - abs(umin)*k
        umax = np.max(targets[i,0,:,:])
        umax = umax + abs(umax)*k
        vmin = np.min(targets[i,1,:,:])
        vmin  = vmin - abs(vmin)*k
        vmax = np.max(targets[i,1,:,:])
        vmax = vmax + abs(vmax)*k
        plt.figure()
        im = plt.imshow(outputs[i,0,:,:], vmin = umin, vmax = umax)
        cbar = plt.colorbar(im)
        cbar.set_label('u (m/s)', rotation=270, labelpad=15)
        plt.title('Inference')
        plt.savefig(os.path.join(output_prefix, f"{plotcount}upred.png"))
        plt.close()

        plt.figure()
        im = plt.imshow(targets[i,0,:,:], vmin = umin, vmax = umax)
        cbar = plt.colorbar(im)
        cbar.set_label('u (m/s)', rotation=270, labelpad=15)
        plt.title('Target')
        plt.savefig(os.path.join(output_prefix, f"{plotcount}utarget.png"))
        plt.close()

        plt.figure()
        im = plt.imshow(outputs[i,1,:,:], vmin = vmin, vmax = vmax)
        cbar = plt.colorbar(im)
        cbar.set_label('v (m/s)', rotation=270, labelpad=15)
        plt.title('Inference')
        plt.savefig(os.path.join(output_prefix, f"{plotcount}vpred.png"))
        plt.close()

        plt.figure()
        im = plt.imshow(targets[i,1,:,:], vmin = vmin, vmax = vmax)
        cbar = plt.colorbar(im)
        cbar.set_label('v (m/s)', rotation=270, labelpad=15)
        plt.title('Target')
        plt.savefig(os.path.join(output_prefix, f"{plotcount}vtarget.png"))
        plt.close()

        plt.figure()
        plt.imshow(inputs[i,1,:,:], cmap='gray')
        plt.savefig(os.path.join(output_prefix, f"{plotcount}im1.png"))
        plt.close()

        plt.figure()
        plt.imshow(inputs[i,0,:,:], cmap='gray')
        plt.savefig(os.path.join(output_prefix, f"{plotcount}im0.png"))
        plt.close()

        plt.figure()
        plt.imshow(masks[i,:,:], cmap='gray')
        plt.savefig(os.path.join(output_prefix, f"{plotcount}mask.png"))
        plt.close()
        plotcount += 1
    
    # Write NetCDF
    # nc_filename = os.path.join(output_prefix, f'results.nc')
    
    # Nt, Nimg, Ny, Nx = inputs.shape
    # varlist = ['img_0','img_1','mask','U_inp', 'V_inp', 'vort_inp', 'div_inp', 'strain_inp',
    #            'U_out', 'V_out', 'vort_out', 'div_out', 'strain_out']
    
    # with ncCreate(nc_filename, Nx, Ny, varlist) as nc:
    #     nc.variables['img_0'][:] = inputs[:,0,:,:]
    #     nc.variables['img_1'][:] = inputs[:,1,:,:]
    #     nc.variables['mask'][:] = masks[:,:,:]
    #     nc.variables['U_inp'][:] = targets[:, 0, :, :]
    #     nc.variables['V_inp'][:] = targets[:, 1, :, :]
    #     nc.variables['U_out'][:] = outputs[:, 0, :, :]
    #     nc.variables['V_out'][:] = outputs[:, 1, :, :]
    #     nc.variables['vort_inp'][:] = true_grads[:, 0, :, :]
    #     nc.variables['div_inp'][:] = true_grads[:, 1, :, :]
    #     nc.variables['strain_inp'][:] = true_grads[:, 2, :, :]
    #     nc.variables['vort_out'][:] = pred_grads[:, 0, :, :]
    #     nc.variables['div_out'][:] = pred_grads[:, 1, :, :]
    #     nc.variables['strain_out'][:] = pred_grads[:, 2, :, :]
        
    #     nc.description = f'Test set results for epoch {epoch}'
    #     nc.input_field = 'input'
    #     nc.output_fields = 'Vorticity, Divergence, Strain (target and predicted)'
    
    # print(f'Test results written to {nc_filename}')


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    # Parse arguments
    args = parse_args()
    args_dict = vars(args)
    args_string = str(args_dict)

    if (args.resume_from_file is not None) or (args.resume_from_idx is not None):
        args.resume = True

    if args.write_log:
        # Start log file
        log_dict = {
            'exp_idx': None,
            'model': args.model,
            'lr': args.lr,
            'batch_size': args.batch_size,
            'c_spec': args.c_spec,
            'use_grad': args.use_grad_loss,
            'epochs': 0,
            'data_root': args.data_root,
            'resume_file': None,
            'rand_trans': str(args.rand_trans),
            'eval_criterion': args.eval_criterion,
            'epoch_best': None,
            'mean': None,
            'r2': None,
            'spec': None,
            'best_model_file': '',
            'args_string':args_string
        }
        columns_list = list(log_dict.keys())
        
        os.makedirs(args.output_dir, exist_ok=True)
        logpath = Path(f"{args.output_dir}/logfile.csv")
        if logpath.is_file():
            print(f"logfile exists at {logpath}")
            logdf = pd.read_csv(logpath, index_col = False)
            print(logdf)
            exp_idx = len(logdf)
            log_dict['exp_idx'] = exp_idx
            for c in range(len(columns_list)):
                if columns_list[c] not in logdf.columns:
                    if c <= len(logdf.columns):
                        logdf.insert(c,columns_list[c],None,allow_duplicates=False)
                    else:
                        logdf.insert(len(logdf.columns),columns_list[c],None,allow_duplicates=False)
                    print(f'Added missing column {columns_list[c]} to the log file')
            
        else:
            print(f"no logfile exists at {logpath}. One will be generated.")
            logdf = pd.DataFrame(columns = columns_list)
            exp_idx = 0
            log_dict['exp_idx'] = exp_idx
            
        logdf = logdf.astype({'resume_file': 'str'})
        logdf = logdf.astype({'data_root': 'str'})
        logdf = logdf.astype({'best_model_file':'str'})
        for k, v in log_dict.items():
            logdf.loc[exp_idx,k] = v
        logdf = logdf.astype({'exp_idx': 'int'})
        logdf.set_index('exp_idx',inplace = True)
        logdf.to_csv(logpath)
        args.exp_idx = exp_idx
        args.logpath = logpath
        args.logdf = logdf

        args.output_subdir = os.path.join(args.output_dir, f'{args.exp_idx}')

    else:
        args.output_subdir = args.output_dir

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.output_subdir, exist_ok=True)

    
    # Setup device
    device = setup_device(args.cuda)
    
    # Determine epochs if not specified
    if args.epochs is None:
        args.epochs = 100 if args.c_spec == 0 else 50
    
    
    # Batch sizes depend on model complexity
    if args.batch_size == None:
        if args.model == 'samudra0' or args.nbase == 32:
            batch_sizes = {'train': 32, 'test': 100, 'valid': 25}
        else:
            batch_sizes = {'train': 64, 'test': 200, 'valid': 50}
    else:
        batch_sizes = {'train': args.batch_size, 'test': 200, 'valid': 50}
    
    if len(args.rand_trans) < 2:
        args.rand_trans = args.rand_trans[0]

    train_data, val_data, test_data = make_splits(
        root=args.data_root,
        subsets=args.subsets,
        ext=args.ext,
        crop_size=args.crop_size,
        train_ratio=0.7,
        val_ratio=0,
        seed=42,
        rand_trans = args.rand_trans,
    )

    train_loader, test_loader = create_dataloaders(train_data, test_data, batch_sizes=batch_sizes)
    
    # Initialize model
    sample_x, sample_y, m = next(iter(test_loader))
    n_input, n_output = sample_x.shape[1], sample_y.shape[1]
    
    sample_x_cpu = sample_x.to("cpu").numpy()[0,:,:,:]
    sample_y_cpu = sample_y.to("cpu").numpy()[0,:,:,:]


    parent_dir = Path(__file__).resolve().parent
    dir = parent_dir / 'debugplots'
    dir.mkdir(exist_ok=True)
    
    plt.figure()
    im = plt.imshow(sample_y_cpu[1,:,:])
    plt.colorbar(im)
    plt.savefig(dir / "y_sample_1.png")
    plt.close()

    plt.figure()
    im = plt.imshow(sample_y_cpu[0,:,:])
    plt.colorbar(im)
    plt.savefig(dir / "y_sample_0.png")
    plt.close()

    plt.figure()
    plt.imshow(sample_x_cpu[1,:,:])
    plt.savefig(dir / "x_sample_1.png")
    plt.close()

    plt.figure()
    plt.imshow(sample_x_cpu[0,:,:])
    plt.savefig(dir / "x_sample_0.png")
    plt.close()

    model_prev = initialize_model(
        n_input, n_output,
        model_name=args.model,
        nbase=args.nbase,
        kernel_size=args.kernel_size,
        device=device
    )
    model = torch.compile(model_prev)
    # Load pretrained weights if using spectral loss or if '--resume' flag is set to true
    model_str = get_model_string(args.model, args.nbase, args.kernel_size, args.use_grad_loss)
    if args.resume: #formerly also triggered by c_spec > 0, but that restricts what you can do a bit.
        if args.write_log and args.resume: #if write_log option is set, find the previous model listed in the log file.
            if args.resume_from_idx:
                stage0_file = f'{logdf.loc[args.resume_from_idx,'best_model_file']}'
            elif args.resume_from_file:
                stage0_file = args.resume_from_file
            elif exp_idx > 0:
                stage0_file = f'{logdf.loc[exp_idx-1,'best_model_file']}'

            if os.path.exists(stage0_file):
                model = load_model(model, stage0_file, device)
                logdf = logdf.astype({'resume_file': 'str'})
                logdf.loc[exp_idx,'resume_file'] = stage0_file
            else:
                print(f'Warning: Stage 0 checkpoint {stage0_file} not found, starting from scratch')

        else: # otherwise just go with the default naming
            stage0_file = f'{model_str}_{args.step0}_{args.nframes}_0.0cs.pth'
            if os.path.exists(stage0_file):
                model = load_model(model, stage0_file, device)
            else:
                print(f'Warning: Stage 0 checkpoint {stage0_file} not found, starting from scratch')
        
    print(logdf)
    logdf.to_csv(logpath)
    args.logdf = logdf

    # Setup optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-5
    )
    
    # Training criterion
    criterion = nn.L1Loss()

    # # Write untrained baseline
    # kernel_x = dx_kernel(args.pm).to(device)
    # kernel_y = dy_kernel(args.pn).to(device)
    # if args.write_log:
    #     output_prefix = args.output_subdir
    # else:
    #     output_prefix = os.path.join(args.output_subdir, f"test_{model_str}_{args.c_spec}cspec")
    # write_test_results(
    #     -1, deepcopy(model), test_loader, kernel_x, kernel_y, output_prefix
    # )
    
    # Train
    best_model, r2_history, mean_history = train_model(
        model, train_loader, test_loader,
        optimizer, criterion, args, device
    )
    
    # Save final results
    if args.write_log:
        np.save(os.path.join(args.output_subdir, 'r2.npy'), r2_history)
        np.save(os.path.join(args.output_subdir, 'mean.npy'), mean_history)
        # save_model(best_model, f'{exp_idx}.pth')
    else:
        np.save(os.path.join(args.output_subdir, f'r2_{model_str}_ver_{args.c_spec}cs.npy'), r2_history)
        np.save(os.path.join(args.output_subdir, f'mean_{model_str}_ver_{args.c_spec}cs.npy'), mean_history)
        # save_model(best_model, f'{model_str}_{args.step0}_{args.nframes}_{args.c_spec}cs.pth')
    
    # # Final satellite inference
    # out_val, grad_val, sst_val = run_satellite_inference(
    #     best_model, args.goes_file, args.valid_inds,
    #     args.pm, args.pn
    # )
    
    # output_file = f'preds_{model_str}_{args.step0}_{args.nframes}_{args.c_spec}cs{args.goes_file}'
    # write_satellite_netcdf(output_file, out_val, grad_val, sst_val, args.valid_inds, args.goes_file)
    
    print('Training complete!')


if __name__ == '__main__':
    main()
