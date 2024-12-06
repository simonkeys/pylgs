"""Extended functionality for [pyMOR](https://pymor.org/) vector arrays"""

# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/api/pymor/vectorarrays.ipynb.

# %% auto 0
__all__ = ['XarrayVectorArrayImpl', 'XarrayVectorArray', 'XarrayVectorSpace', 'plotly_dataarray']

# %% ../../nbs/api/pymor/vectorarrays.ipynb
from numbers import Number, Real
from functools import partial
import inspect
from typing import Optional

from fastcore.basics import patch, first
from fastcore.meta import delegates

import numpy as np
from numpy import array, ndarray
import xarray as xr
from xarray import DataArray, Variable
from pymor.basic import NumpyVectorSpace
from pymor.core.base import classinstancemethod
from pymor.vectorarrays.interface import VectorArray, VectorArrayImpl, VectorSpace

from ..utilities.basic import get_item, filter_args, filter_out_args
from ..utilities.xarray import Coordinates
from ..utilities.formatting import prefix_format

# %% ../../nbs/api/pymor/vectorarrays.ipynb
class XarrayVectorArrayImpl(VectorArrayImpl):

    def __init__(self, _array, space_array):
        self.dims = tuple(set(_array.dims) - set(space_array.dims))
        self.__auto_init(locals())

    @property
    def array(self): return self._array

    @property
    def coords(self): return self._array.coords

    @property
    def _len(self): return self._array.size // self.space_array.size
    
    def __len__(self): return self._len

    def copy(self, deep, ind):
        new_array = self._array if ind is None else self._array[ind].copy()
        return XarrayVectorArrayImpl(new_array, self.space_array)

    @property
    def extended_dims(self):
        return tuple(dim for dim in self._array.dims if dim not in self.space_array.dims)
    
    def to_numpy(self, ensure_copy, ind): raise NotImplementedError
    def real(self, ind): raise NotImplementedError
    def imag(self, ind): raise NotImplementedError
    def conj(self, ind): raise NotImplementedError
    def delete(self, ind): raise NotImplementedError
    def append(self, other, remove_from_other, oind): raise NotImplementedError
    def scal(self, alpha, ind): raise NotImplementedError
    def scal_copy(self, alpha, ind): raise NotImplementedError
    def axpy(self, alpha, x, ind, xind): raise NotImplementedError
    def axpy_copy(self, alpha, x, ind, xind): raise NotImplementedError
    def inner(self, other, ind, oind): raise NotImplementedError
    def pairwise_inner(self, other, ind, oind): raise NotImplementedError
    def lincomb(self, coefficients, ind): raise NotImplementedError
    def norm(self, ind): raise NotImplementedError
    def norm2(self, ind): raise NotImplementedError
    def dofs(self, dof_indices, ind): raise NotImplementedError
    def amax(self, ind): raise NotImplementedError

# %% ../../nbs/api/pymor/vectorarrays.ipynb
def _abbreviate(s):
    s = s.split('(')[0]
    words = s.split()
    if len(words) > 1: return ''.join(w[0].capitalize() for w in words)
    return s[:2]

# %% ../../nbs/api/pymor/vectorarrays.ipynb
def _coords_str(coords, dims=None):
    if not coords: return '{1}'
    if dims is None: dims = list(coords)
    return '{' + ' ⨉ '.join(f'{dim}({len(coords[dim])})' for dim in dims) + '}'

# %% ../../nbs/api/pymor/vectorarrays.ipynb
def _short_coords_str(coords, dims=None):
    if not coords: return '{1}'
    if dims is None: dims = list(coords)
    return '{' + ' ⨉ '.join(f'{_abbreviate(dim)}({len(coords[dim])})' for dim in dims) + '}'

# %% ../../nbs/api/pymor/vectorarrays.ipynb
class XarrayVectorArray(VectorArray):
    """`VectorArray` implementation via xarray arrays."""

    impl_type = XarrayVectorArrayImpl

    @property
    def array(self): return self.impl.array

    @property
    def shape(self): return self.array.shape
    
    @property
    def coords(self): return self.impl.coords

    @property
    def extended_dims(self):
        return self.impl.extended_dims
    
    def __str__(self):
        return str(self.space)[:-1] + ", " + _coords_str(self.coords, self.extended_dims)[1:]
    
    def short_str(self):
        return _short_coords_str(self.space)[:-1] + ", " + _short_coords_str(self.coords, self.extended_dims)[1:]
    
    def _repr_html_(self):
        return str(self)

# %% ../../nbs/api/pymor/vectorarrays.ipynb
class XarrayVectorSpace(VectorSpace):
    """`VectorSpace` of `XarrayVectorArrays`."""

    def __init__(
        self, 
        coords:Optional[Coordinates|dict|DataArray]=None, 
        name:Optional[str]=None, 
        attrs:Optional[dict]=None, 
        id=None      # See `~pymor.vectorarrays.interface.VectorSpace.id`.
    ):
        if isinstance(coords, dict):
            coords = Coordinates(coords)
        elif isinstance(coords, DataArray):
            if name is None: name = coords.name
            coords = Coordinates({dim: coords[dim] for dim in coords.dims})
        # elif isinstance(coords, list):
        #     coords = Coordinates({c.name: c.data for c in coords}) *** Do we need list input?        
        self._array = DataArray(np.zeros(coords.shape if coords is not None else ()), coords=coords, name=name, attrs=attrs) 

        self.dims = self._array.dims
        self.__auto_init(locals())

    @property
    def dim(self): return self._array.size

    @property
    def shape(self): return self._array.shape
    
    @classinstancemethod
    def make_array(cls, obj, id=None):
        vec = cls._array_factory(obj, id=id)
        vec.name = obj.name
        return vec

    @make_array.instancemethod
    def make_array(self, obj):
        """:noindex:""" 
        vec = self._array_factory(obj, space=self)
        vec.name = obj.name
        return vec        

    @classmethod
    def _array_factory(cls, array:DataArray, space):
        return XarrayVectorArray(space, XarrayVectorArrayImpl(array, space._array))

    def zeros(
        self, 
        coords:Optional[dict]=None # Coordinates to extend vector space over
    )->XarrayVectorArray: # `XarrayVectorArray` with all elements equal to zero
        """Return `XarrayVectorArray` of null vectors in XarrayVectorSpace optionally extended to include supplied coordinates `coords`."""
        if coords is None: coords = {}
        array = xr.zeros_like(self._array.expand_dims(coords))
        return self._array_factory(array, space=self)

    def ones(
        self, 
        coords:Optional[dict]=None # Coordinates to extend vector space over
    )->XarrayVectorArray: # `XarrayVectorArray` with all elements equal to one
        """Return `XarrayVectorArray` of vectors with each element equal to one in XarrayVectorSpace optionally extended to include supplied coordinates `coords`."""
        if coords is None: coords = {}
        array = xr.ones_like(self._array.expand_dims(coords))
        return self._array_factory(array, space=self)

    def __str__(self):
        return _coords_str(self.coords)

    def short_str(self):
        return _short_coords_str(self.coords)
    
    def _repr_html_(self):
        return str(self)

# %% ../../nbs/api/pymor/vectorarrays.ipynb
from pymor.vectorarrays.interface import _create_random_values

# %% ../../nbs/api/pymor/vectorarrays.ipynb
@patch
def random(
    self:XarrayVectorSpace, 
    coords=None, 
    distribution='uniform', 
    name=None
)->XarrayVectorArray: # A random `XarrayVectorArray` in the vector space
    array = self._array.expand_dims(coords)
    if name is not None: array = array.rename(name)
    values = _create_random_values(array.shape, distribution)
    array.data = values
    return self.make_array(array)

# %% ../../nbs/api/pymor/vectorarrays.ipynb
@patch(as_prop=True)
def ndim(self:XarrayVectorSpace): 
    """Number of dimensions of the vector space."""
    return self._array.ndim

# %% ../../nbs/api/pymor/vectorarrays.ipynb
@patch
def from_xarray(
    self:XarrayVectorSpace, 
    data:DataArray
)->XarrayVectorArray:
    """Return an `XarrayVectorArray` containing data from `DataArray` `data`."""
    return self.make_array(data)

# %% ../../nbs/api/pymor/vectorarrays.ipynb
@patch
def from_numpy(
    self:XarrayVectorSpace, 
    data:ndarray, 
    coords=None, 
    extended_dim=None, 
    id=None, 
    ensure_copy=False
)->XarrayVectorArray: # A vector array with data from the numpy array
    """Return an `XarrayVectorArray` in the vector space with data from the `ndarray` `data`."""
    if ensure_copy: data = data.copy()
    if data.size == self.dim and extended_dim is None:
        shape = self.shape
        coords = self.coords
        dims = self.dims
    else:
        if coords is None:
            if extended_dim is None: extended_dim = 'len'
            if isinstance(extended_dim, str):
                extended_dim = {extended_dim: np.arange(data.size // self.dim)}
            coords = self.coords.copy()
            coords.update(extended_dim)
            dims = list(extended_dim) + list(self.dims)
            shape = [len(coords[dim]) for dim in dims]
        else: 
            coords = Coordinates(coords)
            dims = coords.dims
            shape = coords.shape
    array = DataArray(
        data.reshape(shape),
        coords=coords,
        dims=dims,
        attrs=self._array.attrs
    )
    return self._array_factory(array, space=self)

# %% ../../nbs/api/pymor/vectorarrays.ipynb
@patch
def rename(
    self:XarrayVectorSpace,
    new_name_or_name_dict=None,
    **names,
):
    return XarrayVectorSpace(self._array.rename(new_name_or_name_dict, **names))

# %% ../../nbs/api/pymor/vectorarrays.ipynb
@patch
def concatenate(self:XarrayVectorSpace, arrays, dim):
    return self.from_xarray(xr.concat([array.array for array in arrays], Variable(first(dim), first(dim.values()))))

# %% ../../nbs/api/pymor/vectorarrays.ipynb
@patch
def __eq__(self:XarrayVectorSpace, other):
    return type(other) is type(self) and self._array.equals(other._array)

# %% ../../nbs/api/pymor/vectorarrays.ipynb
@patch
def __mul__(self:XarrayVectorSpace, other):
    if not isinstance(other, XarrayVectorSpace): raise NotImplementedError
    return XarrayVectorSpace((self._array * other._array).coords, name=other._array.name)

# %% ../../nbs/api/pymor/vectorarrays.ipynb
@patch
def __contains__(self:XarrayVectorSpace, other):
    if isinstance(other, XarrayVectorSpace):
        return self._array.coords.contain(other._array.coords)
    elif isinstance(other, XarrayVectorArray):
        return self._array.coords.contain(other.space._array.coords)
    return False

# %% ../../nbs/api/pymor/vectorarrays.ipynb
@patch
def item(self:XarrayVectorArray):
    """Return the single value stored in the array if there is only one value."""
    return self.array.item()

# %% ../../nbs/api/pymor/vectorarrays.ipynb
@patch(as_prop=True)
def stacked_array(self:XarrayVectorArrayImpl):
    core = {'core': self.space_array.dims} if self.space_array.dims else {}
    extended = {'extended': self.extended_dims} if self.extended_dims else {}
    return self._array.stack(extended | core)

# %% ../../nbs/api/pymor/vectorarrays.ipynb
@patch
def to_numpy(self:XarrayVectorArrayImpl, ensure_copy, ind):
    if ind is not None: raise NotImplementedError
    array = self.stacked_array.to_numpy()
    if not self.space_array.dims: array = array[None, :]
    if not self.extended_dims: array = array[:, None]
    if ensure_copy and not array.flags['OWNDATA']:
        return array.copy()
    else:
        return array

# %% ../../nbs/api/pymor/vectorarrays.ipynb
@patch
def rename(
    self:XarrayVectorArray,
    new_name_or_name_dict=None,
    **names,
):
    """Rename the vector array and/or any of its dimensions."""
    vec = self.space.rename(new_name_or_name_dict, **names).from_xarray(self.array.rename(new_name_or_name_dict, **names))
    if isinstance(new_name_or_name_dict, str): vec.name = new_name_or_name_dict
    return vec

# %% ../../nbs/api/pymor/vectorarrays.ipynb
@patch
def __mul__(self:XarrayVectorArray, other):
    if isinstance(other, DataArray) and self.coords.contain(other.coords): 
        return self.space.from_xarray(self.array * other).rename(self.name)
    elif isinstance(other, XarrayVectorArray): 
        return (self.space * other.space).from_xarray(self.array * other.array).rename(self.name)
    return super().__mul__(other)

# %% ../../nbs/api/pymor/vectorarrays.ipynb
def _reim(da:DataArray):
    return xr.concat([da.real, da.imag], dim=xr.Variable('Part', ['Re', 'Im'])).transpose(*da.dims, 'Part')

# %% ../../nbs/api/pymor/vectorarrays.ipynb
@patch
def real(self:XarrayVectorArrayImpl, ind):
    return XarrayVectorArrayImpl(self.copy(False, ind).array.real, self.space_array)
@patch
def imag(self:XarrayVectorArrayImpl, ind):
    return XarrayVectorArrayImpl(self.copy(False, ind).array.imag, self.space_array)
@patch
def conj(self:XarrayVectorArrayImpl, ind):
    if np.isrealobj(self._array):
        return self.copy(False, ind)
    return XarrayVectorArrayImpl(np.conj(self.copy(False, ind).array), self.space_array)
@patch
def reim(self:XarrayVectorArrayImpl):
    return XarrayVectorArrayImpl(_reim(self.array), self.space_array)
@patch(as_prop=True)
def reim(self:XarrayVectorArray):
    impl = self.impl.reim()
    if impl is self.impl:
        return self.copy()
    else:
        return type(self)(self.space, impl)

# %% ../../nbs/api/pymor/vectorarrays.ipynb
@patch
def scal(self:XarrayVectorArrayImpl, alpha, ind):
    if ind is not None: raise NotImplementedError
    self._array *= alpha

# %% ../../nbs/api/pymor/vectorarrays.ipynb
@patch
def axpy(self:XarrayVectorArrayImpl, alpha, x, ind, xind):
    if ind is not None: raise NotImplementedError
    B = x._array
    if isinstance(alpha, Number):
        if alpha == 1:
            self._array += B
            return
        elif alpha == -1:
            self._array -= B
            return

    self._array += B * alpha

# %% ../../nbs/api/pymor/vectorarrays.ipynb
@patch
def scal_copy(self:XarrayVectorArrayImpl, alpha, ind):
    if ind is not None: raise NotImplementedError

    if isinstance(alpha, Number) and alpha == -1:
        return type(self)(-self._array, self.space_array)

    return type(self)(self._array * alpha, self.space_array)

# %% ../../nbs/api/pymor/vectorarrays.ipynb
@patch
def axpy_copy(self:XarrayVectorArrayImpl, alpha, x, ind, xind):
    if ind is not None: raise NotImplementedError
    B = x._array
    if isinstance(alpha, Number):
        if alpha == 1:
            return type(self)(self._array + B, self.space_array)
        elif alpha == -1:
            return type(self)(self._array - B, self.space_array)
    return type(self)(self._array + B * alpha, self.space_array)

# %% ../../nbs/api/pymor/vectorarrays.ipynb
import pandas as pd
import plotly.express as px
import plotly

# %% ../../nbs/api/pymor/vectorarrays.ipynb
plotly.io.templates.default = "plotly_white"
plotly.io.templates['plotly_white'].layout.legend = plotly.graph_objects.layout.Legend(tracegroupgap=0)
plotly.io.templates["plotly_white"].layout.width = 700
plotly.io.templates["plotly_white"].layout.height = 400
plotly.io.templates["plotly_white"].layout.autosize = False
plotly.io.templates["plotly_white"].layout.colorway = array([
       '#2fa1da', '#fb4f2f', '#e4ae38', '#6d904f', '#8a8a8a', '#16bdcf',
       '#9367bc', '#d62628', '#1f77b3', '#e277c1', '#8c564b', '#bcbc21',
       '#3a0182', '#004200', '#0fffa8', '#5d003f', '#c6bcff', '#424f52',
       '#b80080', '#ffb6b3', '#7c0100', '#6026ff', '#ffff9a', '#aec8aa',
       '#00857c', '#543a00', '#93fbff', '#00bf00', '#7c00a0', '#aa7200',
       '#90ff00', '#01bd89', '#00447b', '#c8826e', '#ff1f82', '#dd00ff',
       '#057400', '#644460', '#878eff', '#ffb5f4', '#526236', '#cd85ff',
       '#676983', '#bdb3bd', '#a56089', '#95d3ff', '#0100f7', '#ff8001',
       '#8a2844', '#aca06d', '#52448a', '#c8ffd8', '#aa4600', '#ff798e',
       '#82d370', '#909ebf', '#9300f4', '#ebcf9a', '#ac8ab1', '#006249',
       '#ffdb00', '#877750', '#7eaaa3', '#000097', '#f400c6', '#643328',
       '#006677', '#03e2c8', '#a736ae', '#c4dbe1', '#4d6eff', '#9a9301',
       '#cd576b', '#efddfd', '#795900', '#5e879a', '#b3ff91', '#5d726b',
       '#520066', '#058750', '#831f6e', '#3b9505', '#647200', '#f0a06b',
       '#5e4f44', '#bc0049', '#cf6726', '#d695aa', '#895dff', '#826b75',
       '#2a54b8', '#6e7cba', '#e6d4d3', '#5d0018', '#7c3b01', '#80b17c',
       '#c8d87c', '#00e83b', '#7cb1ff', '#ff54ff', '#a32621', '#1ce4ff',
       '#7caf3b', '#7b4b90', '#dfff48', '#6b00c3', '#cda897', '#bd62c3',
       '#89cdcd', '#4603c8', '#5d9179', '#414901', '#05a79c', '#cf8c36',
       '#fff7cf', '#425470', '#b544ff', '#cf4993', '#cfa3df', '#93d400',
       '#a793da', '#2da557', '#8ce2b5', '#a3a89c', '#6b5bb6', '#ff7e5d',
       '#a78289', '#afbdd8', '#2ac3ff', '#a5673d', '#f690fd', '#874b64',
       '#ff0c4b', '#215d23', '#4291ff', '#87829c', '#672d44', '#b14f41',
       '#004d52', '#5e1a00', '#ac4167', '#4f3167', '#d6fffd', '#7eb5d1',
       '#a8b869', '#ff95ca', '#c87495', '#364f38', '#ffcf62', '#5d5762',
       '#879375', '#a877ff', '#03c862', '#e6bdd4', '#d4e2cf', '#876790',
       '#897c26', '#cddbff', '#aa676b', '#313474', '#ff5da8', '#009aaf',
       '#70ffdd', '#775b38', '#4f649a', '#cc00b3', '#567b54', '#506e7b',
       '#015e91', '#aabcbd', '#017e99', '#03dd97', '#873a2b', '#ef958e',
       '#75c6aa', '#70695d', '#ccdb08', '#af8556', '#d80075', '#9c3f80',
       '#d84400', '#dd6754', '#5eff79', '#d4b172', '#62265d', '#baa13d',
       '#d8f2b3', '#56018e', '#a19aaa', '#4d4926', '#a3a8ff', '#ace8db',
       '#995901', '#ac00e2', '#46822f', '#cac3ac', '#00c4b5', '#605277',
       '#336d67', '#a59180', '#8399a1', '#fd5664', '#7095d1', '#728c07',
       '#7e004b', '#152fa0', '#d1c1e2', '#c885cf', '#6b444b', '#7e0023',
       '#00a179', '#b1a8cf', '#f90000', '#afe8ff', '#939e4f', '#727982',
       '#d82d54', '#466001', '#0059ff', '#773fb5', '#ace460', '#674424',
       '#525d50', '#957267', '#a8e49a', '#a30057', '#d862f6', '#8e7ccf',
       '#ffbc93', '#a30091', '#9affb8', '#a7c1ff', '#f46200', '#e4efff',
       '#b89ca3', '#609593', '#ff9e34', '#8c2800', '#726b31', '#df824d',
       '#af7bd4', '#bc2d00', '#7b6ea3', '#484262', '#c6a3ff', '#004d28',
       '#c3c68e', '#df48d6', '#e6e864', '#e4c10a', '#00f4f0', '#9e5ba1',
       '#4b41b6', '#64338e', '#757e6b', '#a88936'])

# %% ../../nbs/api/pymor/vectorarrays.ipynb
def _categorical_dims(array:DataArray): return [dim for dim in array.dims if not isinstance(array[dim][0].item(), Number)]
def _numerical_dims(array:DataArray):   return [dim for dim in array.dims if     isinstance(array[dim][0].item(), Number)]

# %% ../../nbs/api/pymor/vectorarrays.ipynb
def _numerical_dim_sort_order(da, dim):
    """Assign priority for each dimension to be put on the x axis.
    Lower number is higher priority. Time is preferred on a scrubber, velocity preferred on the x-axis.
    """
    if "time" in dim.lower(): return 2
    if "velocity" in dim.lower(): return 0
    if isinstance(da[dim][0].item(), Real): return 1
    return 0

# %% ../../nbs/api/pymor/vectorarrays.ipynb
def _set_plotly_frame(fig, n):
    fig.update_layout(sliders=[dict(active=n)])
    for i in range(len(fig.data)):
        fig.data[i].y = fig.frames[n].data[i].y

# %% ../../nbs/api/pymor/vectorarrays.ipynb
def _format_slider_labels(fig, slider_label_precision=2):
    for step in fig.layout.sliders[0].steps:
        step.label = prefix_format(float(step.label), slider_label_precision)

# %% ../../nbs/api/pymor/vectorarrays.ipynb
def plotly_dataarray(da:DataArray, slider_label_precision=2, **kwargs):
    """Visualize the data contained in the `XarrayVectorArray`.
    Put the first dimension with numerical coordinates on the x axis, and include a scrubber if there is a second numerical dimension. 
    Put categorical dimensions in the legend. If the data are complex, add the real and imaginary parts as a categorical dimension.
    If there are no numerical dimensions, plot the categorical dimension(s) as a horizontal bar plot.
    """
    da = da.copy()
    if np.any(np.iscomplex(da)): da = _reim(da)
    if da.name is None: da = da.rename('Value')
    df = da.to_dataframe().reset_index()
    num = _numerical_dims(da)
    cat = _categorical_dims(da)
    if not num and da.ndim <= 2: 
        return px.bar(
            df, 
            y=cat[0], 
            x=da.name, 
            color=get_item(cat, 1), 
            orientation='h', 
            **filter_args(px.bar, **kwargs)
        ).update_layout(
            legend_title=None
        )
    if da.ndim > 4: raise NotImplementedError
    if len(num) > 2: num, cat = num[:2], cat + num[2:]
    if len(cat) > 2: cat, num = cat[:2], num + cat[2:]
    num = sorted(num, key=partial(_numerical_dim_sort_order, da))
    fig = px.line(
        df,
        x=get_item(num, 0),
        y=da.name,
        animation_frame=get_item(num, 1),
        color=get_item(cat, 0),
        line_dash=get_item(cat, 1),
        **filter_args(px.line, **kwargs)
    ).update_layout(
        legend_title=None
    )
    if len(num) == 2:
        padding = .05
        ymin, ymax = da.min(), da.max()
        yrange = (ymax - ymin)
        fig.update_layout(
            yaxis_range=(ymin - padding * yrange, ymax + padding * yrange)
        )
        _format_slider_labels(fig, slider_label_precision=slider_label_precision)
        _set_plotly_frame(fig, len(da[num[1]])//3)
        # Set total animation time to 3s, unless that would make individual frames shorter than 100ms, which seems to be performance limit
        fig.layout.updatemenus[0].buttons[0].args[1]['frame']['duration'] = max(3000/len(da[num[1]]), 100) 
        fig.layout.updatemenus[0].buttons[0].args[1]['transition']['duration'] = 0
    kwargs = filter_out_args(px.line, **kwargs)
    for k in kwargs:
        try: fig.update_layout({k: kwargs[k]})
        except ValueError: pass
        try: fig.update_traces({k: kwargs[k]})
        except ValueError: pass
    return fig

# %% ../../nbs/api/pymor/vectorarrays.ipynb
## Could put plotting hints in the attrs of the coordinates inside the model
@patch
@delegates(plotly_dataarray)
def visualize(self:XarrayVectorArray, **kwargs):
    """Visualize the data contained in the `XarrayVectorArray`.
    Put the first dimension with numerical coordinates on the x axis, and include a scrubber if there is a second numerical dimension.
    Atomic velocity is prioritized to be on the x axis, time is prioritized to be on the scrubber.
    Put categorical dimensions in the legend. If the data are complex, add the real and imaginary parts as a categorical dimension.
    If there are no numerical dimensions, plot the categorical dimension(s) as a horizontal bar plot.
    """
    return plotly_dataarray(self.array, **kwargs)
