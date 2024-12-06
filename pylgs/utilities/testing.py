"""Testing utilities"""

# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/api/utilities/testing.ipynb.

# %% auto 0
__all__ = ['test_array', 'test_da']

# %% ../../nbs/api/utilities/testing.ipynb
import numpy as np
import xarray as xr
from fastcore.test import test_close

# %% ../../nbs/api/utilities/testing.ipynb
def test_array(file, name, arr, update='allow'):
    file = file + '_test.npz'
    try: arrays = dict(np.load(file))
    except FileNotFoundError: 
        if update != 'forbid': arrays = {}
        else: raise
    if name in arrays and update != 'force': test_close(arrays[name], arr)
    elif update != 'forbid':
        print(f'adding {name} array')
        np.savez(file, **(arrays | {name: arr}))
    else: raise RuntimeError(f'Test array {name} not in .npz file {file}.')

# %% ../../nbs/api/utilities/testing.ipynb
def test_da(file, name, da, update='forbid'):
    file = file + '_test.cdf'
    if update != 'force':
        try:
            with xr.open_dataarray(file, group=name) as saved: 
                assert saved.equals(da)
                return
        except AssertionError:
            raise AssertionError(f'Saved DataArray\n{saved}\nnot equal to supplied DataArray\n{da}.')
        except FileNotFoundError: 
            if update != 'allow': raise
        except OSError:
            if update != 'allow': raise RuntimeError(f'Test array {name} not in file {file}.')
    print(f'adding {name} DataArray to {file}')
    da.to_netcdf(file, group=name, mode='a')
