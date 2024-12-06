"""Generic imports for all modules in PyLGS"""

# standard library
from typing import *
from numbers import Number, Integral
import functools
from functools import partial
import re
import string
import warnings
import glob
import logging
import os
import itertools as it
from pathlib import Path
import warnings
from collections import namedtuple

#jupyter
from ipywidgets import widgets
from IPython.display import HTML

# programming tools
from fastcore.basics import ifnone, zip_cycle, listify, flatten, first, patch, patch_to, store_attr
from fastcore.meta import delegates
from plum import dispatch
from pydash import py_, map_values, unzip
import wrapt

# scientific packages
import numpy as np
from numpy import array, ndarray
from pandas import DataFrame, Series
from einops import rearrange, reduce, repeat
import sympy as sy
from sympy.parsing.sympy_parser import parse_expr

#pint
# import pint
# from pint import Quantity
# u = pint.UnitRegistry(force_ndarray_like=True)
# u.default_format = "~P"
# pint.set_application_registry(u)

# scipy
import scipy as sp
import scipy.linalg as spl
from scipy.special import erf, erfinv

# pymor
import pymor
from pymor.basic import Mu, Parameters, ExpressionParameterFunctional, NumpyGenericOperator, NumpyMatrixOperator, LincombOperator, NumpyVectorSpace, StationaryModel, InstationaryModel, GenericParameterFunctional
from pymor.core.base import ImmutableObject
from pymor.parameters.base import ParametricObject
from pymor.algorithms.to_matrix import to_matrix
from pymor.algorithms.simplify import expand, contract

# xarray and friends
import xarray as xr
from xarray import DataArray, Dataset
# import pint_xarray
# import xrft as xf
xr.set_options(keep_attrs=True)

warnings.filterwarnings("ignore", category=FutureWarning)



class Dummy():
    pass