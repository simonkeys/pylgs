"""Extended functionality for [pyMOR](https://pymor.org/) operators"""

# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/api/pymor/operators.ipynb.

# %% ../../nbs/api/pymor/operators.ipynb 2
from __future__ import annotations
from pathlib import Path
import importlib
from numbers import Number
import math
from functools import reduce

from fastcore.basics import patch
from fastcore.meta import delegates
from pydash import unzip

import numpy as np
from numpy import array, ndarray
from scipy.sparse import issparse, sparray
from sparse import SparseArray
import sympy as sy
import pandas as pd
from xarray import DataArray, Coordinates

from pymor.basic import LincombOperator, NumpyMatrixOperator, ZeroOperator, ExpressionParameterFunctional, NumpyVectorSpace, Mu, IdentityOperator
from pymor.models.interface import Model
from pymor.algorithms.to_matrix import to_matrix, ToMatrixRules
from pymor.algorithms.simplify import ExpandRules, expand, ContractRules, contract
from pymor.algorithms.rules import match_class, RuleTable
from pymor.algorithms.lincomb import assemble_lincomb
from pymor.operators.interface import Operator
from pymor.core.base import abstractmethod
from pymor.operators.interface import Operator
from pymor.parameters.functionals import ParameterFunctional

from ..patches import *
from ..utilities.xarray import *
from ..utilities.sparse import sparse, sparse2d, kron, sparse2d_rand
from .parameters import *
from .vectorarrays import *

# %% auto 0
__all__ = ['XarrayMatrixOperator', 'densify', 'XarrayFunctionalOperator', 'SumOperator', 'ScaleOperator', 'ProductOperator']

# %% ../../nbs/api/pymor/operators.ipynb
def _mapping_str(op:Operator):
    return f"{op.source.short_str()} → {op.range.short_str()}"

# %% ../../nbs/api/pymor/operators.ipynb
class XarrayMatrixBasedOperator(Operator):
    """Base class for operators that assemble into an `XarrayMatrixOperator`."""

    linear = True

    @property
    def H(self):
        if not self.parametric:
            return self.assemble().H
        else:
            return super().H

    @abstractmethod
    def assemble(self, mu=None): pass

    def apply(self, U, mu=None):
        return self.range.from_xarray(self.assemble(mu).matrix.dot(U.impl._array))

    def apply_adjoint(self, V, mu=None):
        return self.assemble(mu).apply_adjoint(V)

    def as_range_array(self, mu=None):
        return self.range.from_xarray(self.assemble(mu).matrix.copy())

    def as_source_array(self, mu=None):
        return self.source.from_xarray(self.assemble(mu).matrix.copy()).conj()

    @property
    def display_name(self):
        return self.name if self.name != self.__class__.__name__ else 'X'

    def __str__(self):
        return f"{self.display_name}{_mapping_str(self)}"

    def _repr_html_(self):
        return str(self)

# %% ../../nbs/api/pymor/operators.ipynb
def _ndims(matrix):
    if isinstance(matrix, DataArray):
        return len(matrix.dims)
    else:
        return matrix.ndim    

# %% ../../nbs/api/pymor/operators.ipynb
def _get_space_dims(dims, matrix, default):
    if dims is None:
        if hasattr(matrix, 'dims'): return list(matrix.dims)
        else: return [default]
    elif isinstance(dims, str): return [dims]
    elif hasattr(dims, 'dims'): return dims.dims
    return list(dims)

# %% ../../nbs/api/pymor/operators.ipynb
class XarrayMatrixOperator(XarrayMatrixBasedOperator):
    """An `Operator` backed by an xarray `DataArray`."""

    def __init__(
        self, 
        matrix:DataArray|ndarray|SparseArray|sparray, # N-dimensional matrix
        source:XarrayVectorSpace|Coordinates|dict|list|str=None, # Source vector space, coordinates, or dimension name(s)
        range:XarrayVectorSpace|Coordinates|dict|list|str=None, # Range vector space, coordinates, or dimension name(s)
        solver_options:dict=None, # Options for matrix solver
        name:str=None # Operator name
    ):
        if (source is None or range is None) and _ndims(matrix) != 2: 
            raise ValueError(f'source and range must be specified if array {matrix} does not have two dimensions.')
        if issparse(matrix): matrix = sparse(matrix)
        range_dims = _get_space_dims(range, matrix[:, 0], 'range')
        source_dims = _get_space_dims(source, matrix[0], 'source')
        if not isinstance(matrix, DataArray):
            matrix = DataArray(
                matrix, 
                dims=range_dims + source_dims, 
                coords=getattr(range, 'coords', Coordinates()) + getattr(source, 'coords', Coordinates()),
                name=name
            )
        if name is None: name = matrix.name
        source = XarrayVectorSpace({dim: matrix[dim] for dim in source_dims})
        range = XarrayVectorSpace({dim: matrix[dim] for dim in range_dims}, name=name)
        self.__auto_init(locals())

    def assemble(self, mu=None): return self

    @property
    def size(self):
        return self.matrix.size
        
    @property
    def sparse(self):
        return isinstance(self.matrix.data, SparseArray)

# %% ../../nbs/api/pymor/operators.ipynb
@patch(as_prop=True)
def H(self:XarrayMatrixOperator):
    """The adjoint operator."""
    adjoint_matrix = self.matrix if np.isrealobj(self.matrix) else self.matrix.conj()           
    options = {'inverse': self.solver_options.get('inverse_adjoint'),
               'inverse_adjoint': self.solver_options.get('inverse')} if self.solver_options else None
    return self.with_(matrix=adjoint_matrix, source=self.range, range=self.source, 
                      solver_options=options, name=self.name + '_adjoint')

# %% ../../nbs/api/pymor/operators.ipynb
@patch
def apply(
    self:XarrayMatrixOperator, 
    U, # `XarrayVectorArray` of vectors to which the operator is applied
    mu=None # The parameter values for which to evaluate the operator
)->XarrayVectorArray: # `XarrayVectorArray` in the range `XarrayVectorSpace`
    """Apply the operator to an `XarrayVectorArray` in the source vector space."""
    U = self.range.make_array(self.matrix.dot(U.array))
    if self.name != self.__class__.__name__: U = U.rename(self.name)
    return U

# %% ../../nbs/api/pymor/operators.ipynb
@patch
def apply_adjoint(
    self:XarrayMatrixOperator, 
    V, # `XarrayVectorArray` of vectors to which the operator is applied
    mu=None # The parameter values for which to evaluate the operator
)->XarrayVectorArray: # `XarrayVectorArray` in the source `XarrayVectorSpace`
    """Apply the adjoint of the operator to an `XarrayVectorArray` in the range vector space."""
    return self.H.apply(V, mu=mu)

# %% ../../nbs/api/pymor/operators.ipynb
@patch
def _assemble_lincomb(self:XarrayMatrixOperator, operators, coefficients, identity_shift=0., solver_options=None, name=None):
    if not all(isinstance(op, XarrayMatrixOperator) for op in operators):
        return None

    common_mat_dtype = reduce(np.promote_types,
                              (op.matrix.dtype for op in operators if hasattr(op, 'matrix')))
    common_coef_dtype = reduce(np.promote_types, (type(c) for c in coefficients + [identity_shift]))
    common_dtype = np.promote_types(common_mat_dtype, common_coef_dtype)

    if coefficients[0] == 1:
        matrix = operators[0].matrix.astype(common_dtype)
    else:
        matrix = operators[0].matrix * coefficients[0]
        if matrix.dtype != common_dtype:
            matrix = matrix.astype(common_dtype)

    for op, c in zip(operators[1:], coefficients[1:]):
        if c == 1:
            try:
                matrix += op.matrix
            except NotImplementedError:
                matrix = matrix + op.matrix
        elif c == -1:
            try:
                matrix -= op.matrix
            except NotImplementedError:
                matrix = matrix - op.matrix
        else:
            try:
                matrix += (op.matrix * c)
            except NotImplementedError:
                matrix = matrix + (op.matrix * c)

    if identity_shift: raise NotImplementedError

    return XarrayMatrixOperator(matrix, source=self.source, range=self.range, solver_options=solver_options)

# %% ../../nbs/api/pymor/operators.ipynb
@match_class(XarrayMatrixOperator)
def to_matrix_XarrayMatrixOperator(self, op):
    """Return the operator as a 2D matrix with stacked range and source dimensions in either `ndarray` or `sparray` format."""
    data = op.matrix.stack(_range=op.range.coords.dims, _source=op.source.coords.dims).data
    if isinstance(data, SparseArray): return sparse2d(data)
    return data
    
ToMatrixRules.insert_rule(-1, to_matrix_XarrayMatrixOperator)

# %% ../../nbs/api/pymor/operators.ipynb
@patch
def to_numpy(self:XarrayMatrixOperator):
    """Return the operator in dense `ndarray` format."""
    return self.matrix.to_numpy()

# %% ../../nbs/api/pymor/operators.ipynb
class DensifyRules(RuleTable):
    """|RuleTable| for the :func:`densify` algorithm."""

    def __init__(self):
        super().__init__(use_caching=True)

    @match_class(XarrayMatrixOperator)
    def action_XarrayMatrixOperator(self, op):
        op = self.replace_children(op)
        if op.sparse:
            matrix = op.matrix.copy()
            matrix.data = matrix.data.todense()
            op = op.with_(matrix=matrix)
        return op
    
    @match_class(Model, Operator)
    def action_recurse(self, op):
        return self.replace_children(op)

# %% ../../nbs/api/pymor/operators.ipynb
def densify(
    obj:Model|Operator # Object to densify
)->Model|Operator: # Densified object
    """Convert sparse operators to dense."""
    return DensifyRules().apply(obj)

# %% ../../nbs/api/pymor/operators.ipynb
class XarrayFunctionalOperator(XarrayMatrixBasedOperator):
    """An `Operator` described by a `ParameterFunctional` that assembles to a `XarrayMatrixOperator`."""
    
    def __init__(self, functional:ParameterFunctional, range, source):
        self.__auto_init(locals())

    def assemble(self, mu=None): 
        return XarrayMatrixOperator(self.functional.evaluate(mu), range=self.range, source=self.source)

    @property
    def size(self):
        return self.matrix.size

# %% ../../nbs/api/pymor/operators.ipynb
class SumOperator(Operator):
    """An `Operator` that sums over one or more dimensions of a `XarrayVectorSpace`."""
    
    linear = True
    
    def __init__(
        self, 
        source_coords:Coordinates|dict, # Coordinates to sum over 
        name=None # Operator name
    ):
        self.__auto_init(locals())
        self.source = XarrayVectorSpace(source_coords)
        self.range = XarrayVectorSpace()

    def assemble(self, mu=None): return self

    def apply(self, U, mu=None):
        U = self.range.from_xarray(U.array.sum(self.source.dims))
        if self.name != self.__class__.__name__: U = U.rename(self.name)
        return U

    def __mul__(self, other):
        if isinstance(other, Operator):
            return ProductOperator([self, other])
        return super().__mul__(other)

    def __rmul__(self, other):
        if isinstance(other, Operator):
            return ProductOperator([other, self])
        return super().__mul__(other)

    def __str__(self):
        return f"{self.name if self.name != self.__class__.__name__ else '∑'}{_mapping_str(self)}"

    def _repr_html_(self):
        return str(self)

# %% ../../nbs/api/pymor/operators.ipynb
class ScaleOperator(Operator):
    """A scaling operator for `XarrayVectorSpace`s."""
    
    def __init__(self, array, space=None, name=None):
        if not isinstance(array, DataArray): array = DataArray(array, coords=space.coords, name=name)
        if space is None: space = XarrayVectorSpace(array)
        self.__auto_init(locals())
        self.source = space
        self.range = space

    def assemble(self, mu=None): return self

    def apply(self, U, mu=None):
        U = U * self.array
        if self.name != self.__class__.__name__: U = U.rename(self.name)
        return U

    @property
    def display_name(self):
        return self.name if self.name != self.__class__.__name__ else 'S'
    
    def __mul__(self, other):
        if isinstance(other, Operator):
            return ProductOperator([self, other])
        return super().__mul__(other)

    def __rmul__(self, other):
        if isinstance(other, Operator):
            return ProductOperator([other, self])
        return super().__mul__(other)

    def __str__(self):
        return f"{self.display_name}{_mapping_str(self)}"

    def _repr_html_(self):
        return str(self)

# %% ../../nbs/api/pymor/operators.ipynb
def _no_space_str(op):
    if isinstance(op, XarrayMatrixOperator): return op.name if op.name != op.__class__.__name__ else 'X'
    if isinstance(op, LincombOperator): return '[' + ' + '.join(f'{_no_space_str(c)}·{_no_space_str(o)}' for c, o in zip(op.coefficients, op.operators)) + ']'
    return str(op)

# %% ../../nbs/api/pymor/operators.ipynb
@patch
def __str__(self:LincombOperator):
    return f"{_no_space_str(self)}{_mapping_str(self)}"

@patch
def _repr_html_(self:LincombOperator):
    return str(self)

# %% ../../nbs/api/pymor/operators.ipynb
def _numpy_zero_operator(shape):
    return ZeroOperator(*[NumpyVectorSpace(dim) for dim in shape])

# %% ../../nbs/api/pymor/operators.ipynb
def _read_sparse_dataarray(
    file_name:str|Path, # File name
)->DataArray:
    if not Path(file_name).suffix: file_name = Path(file_name).with_suffix('.mtxn')
    header = []
    with open(file_name, "r") as reader:
        while True:
            line = next(reader)
            if line[0] != "%": break
            header.append(line.strip('%\n '))
    matrix = pd.read_csv(
        file_name,
        sep='\s+',
        header=None,
        skiprows=len(header),
        index_col=None,
        engine="c",
        comment='%'
    )
    name = header[0]
    parameters = header[1].split('\t')
    shape = [int(s) for s in header[2].split()]
    coords = {dim: index.split('\t') for dim, index in zip(header[3::2], header[4::2])}
    matrix = sparse([matrix.iloc[:, :-1].T, matrix.iloc[:, -1]], shape=shape)
    return DataArray(matrix, coords, name=name, attrs={'parameters': parameters})

# %% ../../nbs/api/pymor/operators.ipynb
@patch(cls_method=True)
@delegates(LincombOperator.__init__)
def from_file(
    cls:LincombOperator, 
    file_name:str|Path, # File name
    **kwargs # Additional arguments are passed to `LincombOperator`.
):
    """Read a `LincombOperator` (list of sparse arrays and corresponding `ExpressionParameterFunctionals`) from a .mtxn file."""
    da = _read_sparse_dataarray(file_name)
    coefficients = [
        ExpressionParameterFunctional(s, {p: 1 for p in da.parameters if p in s}) if da.parameters != [''] else 1
        for s in da.coords[da.dims[0]].data
    ]
    del da.attrs['parameters']
    operators = [XarrayMatrixOperator(d.drop_vars(da.dims[0])) for d in da]    
    return LincombOperator(operators, coefficients, **kwargs)

# %% ../../nbs/api/pymor/operators.ipynb
@patch
def partial_evaluate(
    self:ExpressionParameterFunctional, 
    mu:dict|Mu # Parameter values 
)->float|ExpressionParameterFunctional:
    """Substitute parameter values, returning a new `ExpressionParameterFunctional` if expression does not evaluate to a number."""
    mu = {k: v[0] for k, v in mu.items()}
    expr = sy.sympify(self.expression).subs(mu)
    unevaluated = [str(s) for s in expr.free_symbols]
    if expr.is_number:
        return float(expr)
    return ExpressionParameterFunctional(
        str(expr), 
        {k: v for k, v in self.parameters.items() if k in unevaluated}
    )

# %% ../../nbs/api/pymor/operators.ipynb
@patch
def partial_evaluate_coefficients(
    self:LincombOperator, 
    mu:Mu # Parameter values to substitute
):
    """Substitute parameter values into linear coefficients, returning a new `ExpressionParameterFunctional` if expression does not evaluate to a number."""
    result = []
    for c in self.coefficients:
        try:
            result.append(c.evaluate(mu))
        except AttributeError:
            result.append(c)
        except AssertionError:
            try:
                result.append(c.partial_evaluate(mu))
            except Exception as e:
                result.append(c)
    return result

# %% ../../nbs/api/pymor/operators.ipynb
@patch
def partial_assemble(
    self:LincombOperator, 
    mu=None # Parameter values to substitute
)->Operator:
    """Substitute parameter values into the linear coefficients, returning a new operator with fewer (or no) parameters."""
    operators = tuple(op.assemble(mu) for op in self.operators)
    coefficients = self.partial_evaluate_coefficients(mu)
    evaluated = [[], []]
    unevaluated = [[], []]
    for o, c in zip(operators, coefficients):
        if isinstance(c, Number):
            if c:
                evaluated[0].append(o)
                evaluated[1].append(c)
        else:
            unevaluated[0].append(o)
            unevaluated[1].append(c)
    op = []
    if len(evaluated[0]):
        op.append(assemble_lincomb(
            *evaluated, 
            solver_options=self.solver_options,
            name=self.name + '_partially_assembled'
        ))
    if len(unevaluated[0]):
        op.append(LincombOperator(*unevaluated))
    if len(op) == 0:
        op = [ZeroOperator(range=self.range, source=self.source)]
    op = np.sum(op)
    # To avoid infinite recursions, only use the result if at least one of the following
    # is true:
    #   - The operator is parametric, so the the result of assemble *must* be a different,
    #     non-parametric operator.
    #   - One of self.operators changed by calling 'assemble' on it.
    #   - The result of assemble_lincomb is of a different type than the original operator.
    #   - assemble_lincomb could simplify the list of assembled operators,
    #     which we define to be the case when the number of operators has ben reduced.
    if (self.parametric
            or operators != self.operators  # for this comparison to work self.operators always has to be a tuple!
            or type(op) != type(self)
            or len(op.operators) < len(operators)):
        return op
    else:
        return self

# %% ../../nbs/api/pymor/operators.ipynb
@patch(as_prop=True)
def terms(self:LincombOperator):
    return [c * o for c, o in zip(self.coefficients, self.operators)]

# %% ../../nbs/api/pymor/operators.ipynb
class ProductOperator(Operator):
    """An Operator given by the direct product of operators `operators`."""
    def __init__(
        self, 
        operators, # Sequence of operators that each assemble to XarrayMatrixOperator
        name=None, # Optional name
    ):
        for op in operators:
            if not isinstance(op, Operator): raise ValueError(f'Input {op} should be an Operator.')
        self.source = np.product([op.source for op in operators])
        self.range = np.product([op.range for op in operators])
        self.__auto_init(locals())
        self.linear = True
        
    def apply(self, U, mu=None):
        pass

# %% ../../nbs/api/pymor/operators.ipynb
@patch
def __mul__(self:Operator, other):
    if isinstance(other, Operator) and all(isinstance(space, XarrayVectorSpace) for space in [self.source, self.range, other.source, other.range]):
        return ProductOperator([self, other])
    assert isinstance(other, (Number, ParameterFunctional))
    # from pymor.operators.constructions import LincombOperator
    if self.name != 'LincombOperator' or not isinstance(self, LincombOperator):
        return LincombOperator((self,), (other,))
    else:
        return self.with_(coefficients=tuple(c * other for c in self.coefficients))

# %% ../../nbs/api/pymor/operators.ipynb
@patch
def __str__(self:ProductOperator):
    return ' ⨂ '.join(str(o) for o in self.operators)

# %% ../../nbs/api/pymor/operators.ipynb
@patch
def _repr_html_(self:ProductOperator):
    return str(self)

# %% ../../nbs/api/pymor/operators.ipynb
@patch
def __str__(self:IdentityOperator):
    return f'I{_mapping_str(self)}'

@patch
def _repr_html_(self:IdentityOperator):
    return str(self)

# %% ../../nbs/api/pymor/operators.ipynb
@patch
def apply(self:ProductOperator, U, mu=None):
    U = U.copy()
    for op in self.operators[::-1]:
        U = op.assemble(mu).apply(U)
    return self.range.from_xarray(U.array)

# %% ../../nbs/api/pymor/operators.ipynb
@patch(as_prop=True)
def H(self:ProductOperator):
    return self.with_(operators=[op.H for op in self.operators])

# %% ../../nbs/api/pymor/operators.ipynb
@patch
def apply_adjoint(self:ProductOperator, V, mu=None):
    return self.H.apply(V, mu=mu)

# %% ../../nbs/api/pymor/operators.ipynb
@patch
def assemble(self:ProductOperator, mu=None):
    operators = [op.assemble(mu) for op in self.operators]
    if any(isinstance(op, ZeroOperator) for op in operators): return ZeroOperator(self.range, self.source)
    if all(op == old_op for op, old_op in zip(operators, self.operators)): return self 
    return self.with_(operators=operators)

# %% ../../nbs/api/pymor/operators.ipynb
@match_class(ProductOperator)
def expand_ProductOperator(self, op):
    op = self.replace_children(op)

    # merge child ProductOperators
    if any(isinstance(o, ProductOperator) for o in op.operators):
        ops = []
        for o in op.operators:
            if isinstance(o, ProductOperator):
                ops.extend(o.operators)
            else:
                ops.append(o)
        op = op.with_(operators=ops)

    # expand products with LincombOperators
    if any(isinstance(o, LincombOperator) for o in op.operators):
        i = next(iter(i for i, o in enumerate(op.operators) if isinstance(o, LincombOperator)))
        left, right = op.operators[:i], op.operators[i+1:]
        ops = [ProductOperator(left + (o,) + right) for o in op.operators[i].operators]
        op = op.operators[i].with_(operators=ops)

        # there can still be LincombOperators within the summands so we recurse ..
        op = self.apply(op)
    return op

#| export
ExpandRules.insert_rule(-1, expand_ProductOperator)

# %% ../../nbs/api/pymor/operators.ipynb
@match_class(ProductOperator)
def contract_ProductOperator(self, op):
    op = self.replace_children(op)
    if any(isinstance(o, ZeroOperator) for o in op.operators): return ZeroOperator(op.range, op.source)
    xarray_operators = [o for o in op.operators if     isinstance(o, XarrayMatrixOperator)]
    other_operators  = [o for o in op.operators if not isinstance(o, XarrayMatrixOperator)]
    if xarray_operators:
        matrices, ranges, sources = unzip([(o.matrix, o.range, o.source) for o in xarray_operators])
        op = XarrayMatrixOperator(
            math.prod(matrices), 
            source=math.prod(sources, start=XarrayVectorSpace()), 
            range=math.prod(ranges, start=XarrayVectorSpace())
        )
    else:
        op = 1
    if other_operators:
        op *= math.prod(other_operators) 
    return op

ContractRules.insert_rule(-2, contract_ProductOperator)

# %% ../../nbs/api/pymor/operators.ipynb
@match_class(ProductOperator)
def to_matrix_ProductOperator(self, op):
    return kron(*[self.apply(o) for o in op.operators])
    
ToMatrixRules.insert_rule(-2, to_matrix_ProductOperator)

# %% ../../nbs/api/pymor/operators.ipynb
@patch
def as_source_array(self:ProductOperator, mu=None):
    return contract(self.assemble(mu)).as_source_array()

# %% ../../nbs/api/pymor/operators.ipynb
@patch
def as_range_array(self:ProductOperator, mu=None):
    return contract(self.assemble(mu)).as_range_array()
