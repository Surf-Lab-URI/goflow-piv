"""
Training Utilities
==================
Utility functions for GOFLOW model training, including:
- Model introspection and parameter counting
- Learning rate scheduling (SGDR with cosine annealing)
- Checkpoint management
- Data splitting utilities
- Batch normalization utilities
- NetCDF output helpers

Author: Kaushik Srinivasan (UCLA Atmospheric and Oceanic Sciences)
"""

import os
import math
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# =============================================================================
# Model Introspection
# =============================================================================

def prnModelSt(model: torch.nn.Module):
    """Print model state dict with parameter counts and tensor sizes."""
    print('Number of parameters: %f million' % (nparams(model) / 1e6))
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())


def prnOptimSt(optim: torch.optim.Optimizer):
    """Print optimizer state dict."""
    print("Optimizer's state_dict:")
    for var_name in optim.state_dict():
        print(var_name, "\t", optim.state_dict()[var_name])


def nparams(model: torch.nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# =============================================================================
# Learning Rate Scheduling
# =============================================================================

def setLR(optimizer: torch.optim.Optimizer, lr: float):
    """Set learning rate for all parameter groups in optimizer."""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def wdstep(optimizer: torch.optim.Optimizer, wd: float):
    """Apply weight decay step to all parameters."""
    for group in optimizer.param_groups:
        for param in group['params']:
            param.data = param.data.add(-wd * group['lr'], param.data)


# =============================================================================
# Checkpoint Management
# =============================================================================

def save_checkpoint(dir: str, epoch: int, **kwargs):
    """
    Save training checkpoint with epoch and additional state.

    Args:
        dir: Directory to save checkpoint
        epoch: Current epoch number
        **kwargs: Additional state to save (model, optimizer, etc.)
    """
    state = {
        'epoch': epoch,
    }
    state.update(kwargs)
    filepath = os.path.join(dir, 'checkpoint-%d.pt' % epoch)
    torch.save(state, filepath)


# =============================================================================
# Data Splitting
# =============================================================================

def testTrainSplit(dataset: Dataset, fac: float = 0.8) -> tuple:
    """
    Split dataset into train and test subsets.

    Args:
        dataset: PyTorch dataset to split
        fac: Fraction of data for training (default 0.8)

    Returns:
        Tuple of (train_subset, test_subset)
    """
    train_sz = int(fac * len(dataset))
    test_sz = len(dataset) - train_sz
    train, test = torch.utils.data.random_split(dataset, [train_sz, test_sz])
    return train, test


def testTrainLoader(
    dataset: Dataset,
    batch_size: int = 200,
    fac: float = 0.8
) -> tuple:
    """
    Split dataset and create DataLoaders for train and test.

    Args:
        dataset: PyTorch dataset to split
        batch_size: Batch size for both loaders
        fac: Fraction of data for training (default 0.8)

    Returns:
        Tuple of (train_loader, test_loader)
    """
    train_sz = int(fac * len(dataset))
    test_sz = len(dataset) - train_sz
    train, test = torch.utils.data.random_split(dataset, [train_sz, test_sz])
    train_dl = DataLoader(train, batch_size=batch_size, num_workers=12, shuffle=True)
    test_dl = DataLoader(test, batch_size=batch_size, num_workers=12, shuffle=False)
    return train_dl, test_dl


# =============================================================================
# SGDR Learning Rate Scheduling (Cosine Annealing with Warm Restarts)
# =============================================================================

def TiTcur_ap(epoch: int, T0: int) -> tuple:
    """
    Compute cycle length and start for arithmetic progression scheme.

    In this scheme, cycle lengths increase linearly (T0, 2*T0, 3*T0, ...).

    Args:
        epoch: Current epoch
        T0: Initial cycle length

    Returns:
        Tuple of (current_cycle_length, cycle_start_epoch)
    """
    alpha = 2 * epoch / T0
    index = np.floor((1 + (1 + 4 * alpha) ** 0.5) / 2.0) - 1
    return T0 * (index + 1), T0 * index * (index + 1) / 2


def TiTcur_c(epoch: int, T0: int) -> tuple:
    """
    Compute cycle length and start for constant cycle scheme.

    In this scheme, all cycles have the same length T0.

    Args:
        epoch: Current epoch
        T0: Cycle length

    Returns:
        Tuple of (cycle_length, cycle_start_epoch)
    """
    ti = np.floor(epoch / float(T0)) * T0
    return T0, ti


def cosineSGDR(
    optimizer: torch.optim.Optimizer,
    epoch: int,
    T0: int = 5,
    eta_min: float = 0.0,
    eta_max: float = 0.1,
    scheme: str = 'constant'
) -> float:
    """
    Apply SGDR (Stochastic Gradient Descent with Warm Restarts) learning rate.

    Implements cosine annealing with warm restarts as described in
    "SGDR: Stochastic Gradient Descent with Warm Restarts" (Loshchilov & Hutter).

    Args:
        optimizer: PyTorch optimizer
        epoch: Current epoch
        T0: Initial/base cycle length
        eta_min: Minimum learning rate
        eta_max: Maximum learning rate
        scheme: 'constant' for fixed cycle length, 'linear' for increasing cycles

    Returns:
        Current learning rate
    """
    if scheme == 'linear':
        Ti, ti = TiTcur_ap(epoch, T0)
    else:
        Ti, ti = TiTcur_c(epoch, T0)
    lr = eta_min + 0.5 * (eta_max - eta_min) * (1 + np.cos(np.pi * (epoch - ti) / Ti))
    setLR(optimizer, lr)
    return lr


# =============================================================================
# NetCDF Utilities
# =============================================================================

from netCDF4 import Dataset


def ncCreate(fname: str, Nx: int, Ny: int, varlist: list):
    """
    Create a NetCDF file with specified dimensions and variables.

    Note: This version uses 'x', 'y' dimension names. For 'lon', 'lat' names,
    use writenc.ncCreate instead.

    Args:
        fname: Output filename
        Nx: X dimension size
        Ny: Y dimension size
        varlist: List of variable names to create

    Returns:
        NetCDF Dataset object (caller must close)
    """
    nco = Dataset(fname, 'w')
    nco.createDimension('time', None)
    nco.createDimension('x', Nx)
    nco.createDimension('y', Ny)
    for var in varlist:
        nco.createVariable(var, np.dtype('float32').char, ('time', 'x', 'y'))
    return nco


def addVal(nco, var: str, val, itime: int = None):
    """
    Add values to a NetCDF variable.

    Args:
        nco: NetCDF Dataset object
        var: Variable name
        val: Values to add
        itime: Time index (None to write all time steps)
    """
    if itime is not None:
        nco.variables[var][itime, :, :] = val.squeeze()
    else:
        nco.variables[var][:, :, :] = val.squeeze()


# =============================================================================
# Model Averaging and Perturbation
# =============================================================================

def moving_average(net1: torch.nn.Module, net2: torch.nn.Module, alpha: float = 1):
    """
    Compute exponential moving average of model parameters.

    Updates net1 parameters: net1 = (1-alpha)*net1 + alpha*net2

    Args:
        net1: Target model (will be modified in-place)
        net2: Source model
        alpha: Interpolation factor (1 = copy net2, 0 = keep net1)
    """
    for param1, param2 in zip(net1.parameters(), net2.parameters()):
        param1.data *= (1.0 - alpha)
        param1.data += param2.data * alpha


# =============================================================================
# Batch Normalization Utilities
# =============================================================================

def _check_bn(module: torch.nn.Module, flag: list):
    """Helper to check if module is a BatchNorm layer."""
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        flag[0] = True


def check_bn(model: torch.nn.Module) -> bool:
    """Check if model contains any BatchNorm layers."""
    flag = [False]
    model.apply(lambda module: _check_bn(module, flag))
    return flag[0]


def reset_bn(module: torch.nn.Module):
    """Reset BatchNorm running statistics to initial state."""
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.running_mean = torch.zeros_like(module.running_mean)
        module.running_var = torch.ones_like(module.running_var)


def _get_momenta(module: torch.nn.Module, momenta: dict):
    """Helper to collect BatchNorm momentum values."""
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        momenta[module] = module.momentum


def _set_momenta(module: torch.nn.Module, momenta: dict):
    """Helper to restore BatchNorm momentum values."""
    if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
        module.momentum = momenta[module]


# =============================================================================
# Model Perturbation (for sharpness analysis)
# =============================================================================

def add_gauss_perturbation(module: torch.nn.Module, alpha: float = 1e-4):
    """Add Gaussian noise perturbation to module weights."""
    sample = np.random.normal(0, alpha)
    perturbation = (sample * torch.ones(module.weight.data.shape)).cuda()
    module.weight.data = module.weight.data + perturbation


def model_sharpness(model: torch.nn.Module):
    """
    Add Gaussian perturbations to all Linear and Conv layers recursively.

    Used for analyzing loss landscape sharpness around current parameters.
    """
    for child in model.children():
        module_name = child._get_name()
        if module_name in ['Linear', 'Conv1d', 'Conv2d', 'Conv3d']:
            add_gauss_perturbation(child)
        else:
            model_sharpness(child)


# =============================================================================
# Grid Utilities (ROMS compatibility)
# =============================================================================

def gridDict(dr: str, gridfile: str, ij: tuple = None) -> dict:
    """
    Load grid information from a ROMS grid file.

    Args:
        dr: Directory containing grid file
        gridfile: Grid filename
        ij: Optional subset indices (imin, imax, jmin, jmax)

    Returns:
        Dictionary with grid properties (lon, lat, pm, pn, dx, dy, f, h, mask)
    """
    from os.path import join as pjoin

    def Forder(x):
        return np.asfortranarray(x)

    gdict = {}
    print('gridfile: ' + pjoin(dr, gridfile))

    if ij is not None:
        imin, imax, jmin, jmax = ij[0], ij[1], ij[2], ij[3]
    else:
        imin, imax, jmin, jmax = 0, None, 0, None

    nc = Dataset(pjoin(dr, gridfile), 'r')
    try:
        gdict['lon_rho'] = Forder(nc.variables['lon_rho'][jmin:jmax, imin:imax])
        gdict['lat_rho'] = Forder(nc.variables['lat_rho'][jmin:jmax, imin:imax])
    except:
        gdict['lon_rho'] = Forder(nc.variables['x_rho'][jmin:jmax, imin:imax])
        gdict['lat_rho'] = Forder(nc.variables['y_rho'][jmin:jmax, imin:imax])

    try:
        gdict['x_rho'] = Forder(nc.variables['x_rho'][jmin:jmax, imin:imax])
        gdict['y_rho'] = Forder(nc.variables['y_rho'][jmin:jmax, imin:imax])
    except:
        pass

    gdict['Ny'] = gdict['lon_rho'].shape[1]
    gdict['Nx'] = gdict['lon_rho'].shape[0]
    gdict['pm'] = Forder(nc.variables['pm'][jmin:jmax, imin:imax])
    gdict['pn'] = Forder(nc.variables['pn'][jmin:jmax, imin:imax])
    gdict['dx'] = 1.0 / np.average(gdict['pm'])
    gdict['dy'] = 1.0 / np.average(gdict['pn'])
    gdict['f'] = Forder(nc.variables['f'][jmin:jmax, imin:imax])

    try:
        gdict['h'] = Forder(nc.variables['h'][jmin:jmax, imin:imax])
    except:
        print('No h in grid file')
    try:
        gdict['rho0'] = nc.rho0
    except:
        print('No rho0 in grid file')
    try:
        gdict['mask_rho'] = Forder(nc.variables['mask_rho'][jmin:jmax, imin:imax])
    except:
        print('No mask in file')

    nc.close()
    print(gdict['Nx'], gdict['Ny'])
    return gdict


# =============================================================================
# System Utilities
# =============================================================================

import psutil


def getRam():
    """Print current process memory usage in GB."""
    pid = os.getpid()
    py = psutil.Process(pid)
    memory_use = py.memory_info()[0] / 2.**30
    print(f"memory use: {memory_use} GB")


# ============================================================================
# Flow File Utilities
# ============================================================================

import os, io, time, re
from glob import glob
from typing import Union, List, Tuple, Optional

TAG_STRING = 'PIEH'
TAG_FLOAT = 202021.25
flags = {
    'debug': False
}

"""
flowIO.h
"""
UNKNOWN_FLOW_THRESH = 1e9

def array_cropper(array, crop_window: Union[int, Tuple[int, int, int, int]] = 0):
    # Cropper init.
    s = array.shape  # Create image cropper
    crop_window = (crop_window,) * 4 if type(crop_window) is int else crop_window
    assert len(crop_window) == 4

    # Cropping the array
    return array[crop_window[0] : s[0]-crop_window[1], crop_window[2] : s[1]-crop_window[3]]

def read_flow(filename: str, use_stereo: bool = False,
              crop_window: Union[int, Tuple[int, int, int, int]] = 0) -> np.array:
    """
        Read a .flo file (Middlebury format).
        Parameters
        ----------
        filename : str
            Filename where the flow will be read. Must have extension .flo.
        Returns
        -------
        flow : ndarray, shape (height, width, 2), dtype float32
            The read flow from the input file.
        """

    if not isinstance(filename, io.BufferedReader):
        if not isinstance(filename, str):
            raise AssertionError(
                "Input [{p}] is not a string".format(p=filename))
        if not os.path.isfile(filename):
            raise AssertionError(
                "Path [{p}] does not exist".format(p=filename))
        if not filename.split('.')[-1] == 'flo':
            raise AssertionError(
                "File extension [flo] required, [{f}] given".format(f=filename.split('.')[-1]))

        flo = open(filename, 'rb')
    else:
        flo = filename

    tag = np.frombuffer(flo.read(4), np.float32, count=1)[0]
    if not TAG_FLOAT == tag:
        raise AssertionError("Wrong Tag [{t}]".format(t=tag))

    width = np.frombuffer(flo.read(4), np.int32, count=1)[0]
    if not (width > 0 and width < 100000):
        raise AssertionError("Illegal width [{w}]".format(w=width))

    height = np.frombuffer(flo.read(4), np.int32, count=1)[0]
    if not (width > 0 and width < 100000):
        raise AssertionError("Illegal height [{h}]".format(h=height))

    n_bands = 3 if use_stereo else 2
    size = n_bands * width * height
    tmp = np.frombuffer(flo.read(n_bands * width * height * 4), np.float32, count=size)
    flow = np.resize(tmp, (int(height), int(width), int(n_bands)))
    flo.close()

    return array_cropper(flow, crop_window=crop_window)


def read_flow_collection(dirname: str, start_at: int = 0, num_images: int = -1, use_stereo: bool = False,
                         crop_window: Union[int, Tuple[int, int, int, int]] = 0) -> Tuple[np.array, List[str]]:
    """
    Load a collection of .flo files.
    An example directory may look like:
        dirname
            - frame_0001.flo
            - frame_0002.flo
            - ...
    Parameters
    ----------
    dirname : str
        Directory containing .flo files.
    Returns
    -------
    flows : ndarray, shape (N, H, W, 2)
        Sequence of flow components.
    """
    pattern = re.compile('\d+')

    files = []

    allfiles = [f for f in os.listdir(dirname) if f.endswith('.flo')]
    for f in allfiles:
        match = pattern.findall(f)
        if len(match) > 0:
            frame_index = int(match[-1])
            filepath = os.path.join(dirname, f)
            files.append((frame_index, filepath))

    files = sorted(files, key=lambda x: x[0])
    files_sliced = files[start_at:] if num_images < 0 else files[start_at:start_at+num_images]

    flos, flonames = [], []
    for frame_index, filepath in files_sliced:
        flo_frame = read_flow(filepath, use_stereo=use_stereo,crop_window=crop_window)
        flos.append(flo_frame)
        flonames.append(filepath)

    flos = np.array(flos)

    return flos, flonames


def write_flow(flow: np.ndarray, filename: str, norm: bool = False):
    """
    Write a .flo file (Middlebury format).
    Parameters
    ----------
    flow : ndarray, shape (height, width, 2), dtype float32
        Flow to save to file.
    filename : str
        Filename where flow will be saved. Must have extension .flo.
    norm : bool
        Logical option to normalize the input flow or not.
    Returns
    -------
    None
    """

    assert type(filename) is str, "file is not str (%r)" % str(filename)
    assert filename[-4:] == '.flo', "file ending is not .flo (%r)" % filename[-4:]

    height, width, n_bands = flow.shape
    assert n_bands == 2 or n_bands == 3, "Number of bands = %r != (2 or 3)" % n_bands

    # Extract the u and v velocity
    if norm:  # use flow normalization
        u, v = _normalize_flow(flow)
    else:
        u = flow[:, :, 0]
        v = flow[:, :, 1]
        w = flow[:, :, 2] if flow.shape[2] > 2 else None

    assert u.shape == v.shape, "Invalid flow shape"
    height, width = u.shape

    with open(filename, 'wb') as f:
        tag = np.array([TAG_FLOAT], dtype=np.float32)  # assign ASCCII tag for float 202021.25 (TAG_FLOAT)
        tag.tofile(f)
        np.array([width], dtype=np.int32).astype(np.int32).tofile(f)  # assign width size to ASCII
        np.array([height], dtype=np.int32).tofile(f)  # assign height size to ASCII
        flow.tofile(f)  # assign the array value (u, v)


def resize_flow(flow, des_width, des_height, method='bilinear'):
    """Utility function to resize the flow array, used by RandomScale transformer.
    WARNING: improper for sparse flow!
    Args:
        flow: the flow array
        des_width: Target width
        des_height: Target height
        method: interpolation method to resize the flow
    Returns:
        the resized flow
    """
    src_height = flow.shape[0]
    src_width = flow.shape[1]

    if src_width == des_width and src_height == des_height:  # Sanity check, if resizing is a necessary
        return flow

    ratio_height = float(des_height) / float(src_height)
    ratio_width = float(des_width) / float(src_width)

    if method == 'bilinear':
        flow = cv2.resize(flow, (des_width, des_height), interpolation=cv2.INTER_LINEAR)
    elif method == 'nearest':
        flow = cv2.resize(flow, (des_width, des_height), interpolation=cv2.INTER_NEAREST)
    else:
        raise Exception('Invalid resize flow method!')

    flow[:, :, 0] = flow[:, :, 0] * ratio_width
    flow[:, :, 1] = flow[:, :, 1] * ratio_height

    return flow


def horizontal_flip_flow(flow):
    flow = np.copy(np.fliplr(flow))
    flow[:, :, 0] *= -1
    return flow


def vertical_flip_flow(flow):
    flow = np.copy(np.flipud(flow))
    flow[:, :, 1] *= -1
    return flow
