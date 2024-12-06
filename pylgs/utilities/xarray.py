"""Extra functionality for [xarray](https://xarray.pydata.org/)"""

# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/api/utilities/xarray.ipynb.

# %% auto 0
__all__ = []

# %% ../../nbs/api/utilities/xarray.ipynb
from fastcore.basics import patch
from xarray import Coordinates
import numpy as np

# %% ../../nbs/api/utilities/xarray.ipynb
@patch
def __add__(self:Coordinates, other):
    """Adding two `Coordinates` objects combines their coordinates."""
    if not isinstance(other, Coordinates): raise TypeError
    result = self.copy()
    result.update(other)
    return result

# %% ../../nbs/api/utilities/xarray.ipynb
@patch
def complement(self:Coordinates, other:Coordinates):
    """Return coordinates not in other coordinates."""
    return Coordinates({k: v for k, v in self.items() if k not in other})

# %% ../../nbs/api/utilities/xarray.ipynb
@patch(as_prop=True)
def shape(self:Coordinates):
    """Return tuple of sizes of the coordinates."""
    return tuple(self.sizes.values())

# %% ../../nbs/api/utilities/xarray.ipynb
@patch(as_prop=True)
def size(self:Coordinates):
    """Return product of coordinate lengths."""
    return np.prod(self.shape)

# %% ../../nbs/api/utilities/xarray.ipynb
@patch
def intersection(self:Coordinates, other:Coordinates):
    """Return coordinates in self and other."""
    return Coordinates({k: v for k, v in self.items() if k in other and v.equals(other[k])})

# %% ../../nbs/api/utilities/xarray.ipynb
@patch
def contain(self:Coordinates, other:Coordinates):
    """Return true if all coordinates in `other` are in `self`, otherwise false."""
    return self.intersection(other).equals(other)