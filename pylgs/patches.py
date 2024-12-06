
from __future__ import annotations
from fastcore.basics import patch

### Fix bug in AssembleLincombRules -

from pymor.algorithms.rules import match_always
from pymor.algorithms.lincomb import AssembleLincombRules
from pymor.operators.constructions import LincombOperator


@match_always
def action_return_lincomb(self, ops):
    return LincombOperator(ops, self.coefficients, name=self.name, solver_options=self.solver_options)

try:
    i = [rule.action.__name__ for rule in AssembleLincombRules].index('action_return_lincomb')
    AssembleLincombRules.rules.pop(i)
except: pass
AssembleLincombRules.append_rule(action_return_lincomb)

### TimeStepper.solve -


from pymor.algorithms.timestepping import TimeStepper


@patch
def solve(self:TimeStepper, initial_time, end_time, initial_data, operator, rhs=None, mass=None, mu=None, num_values=None):
    """Apply time-stepper to the equation.

    The equation is of the form ::

        M(mu) * d_t u + A(u, mu, t) = F(mu, t),
                         u(mu, t_0) = u_0(mu).

    Parameters
    ----------
    initial_time
        The time at which to begin time-stepping.
    end_time
        The time until which to perform time-stepping.
    initial_data
        The solution vector at `initial_time`.
    operator
        The |Operator| A.
    rhs
        The right-hand side F (either |VectorArray| of length 1 or |Operator| with
        `source.dim == 1`). If `None`, zero right-hand side is assumed.
    mass
        The |Operator| M. If `None`, the identity operator is assumed.
    mu
        |Parameter values| for which `operator` and `rhs` are evaluated. The current
        time is added to `mu` with key `t`.
    num_values
        The number of returned vectors of the solution trajectory. If `None`, each
        intermediate vector that is calculated is returned.

    Returns
    -------
    |VectorArray| containing the solution trajectory.
    """
    try:
        num_time_steps = self.estimate_time_step_count(initial_time, end_time)
    except NotImplementedError:
        num_time_steps = 0
    iterator = self.iterate(initial_time, end_time, initial_data, operator, rhs=rhs, mass=mass, mu=mu,
                            num_values=num_values)
    U = operator.source.empty(reserve=num_values if num_values else num_time_steps + 1)
    t = []
    for U_n, t_n in iterator:
        U.append(U_n)
        t.append(t_n)
    return U, t

### Add scipy_lgmres_spilu solver method to pymor.bindings.scipy


from scipy.sparse import csr_array
from pymor.core.defaults import defaults
import pymor.bindings.scipy
import numpy as np
from scipy.linalg import solve, solve_continuous_are, solve_continuous_lyapunov, solve_discrete_lyapunov
from scipy.sparse.linalg import LinearOperator, bicgstab, lgmres, lsqr, spilu, splu, spsolve

from pymor.algorithms.genericsolvers import _parse_options
from pymor.algorithms.lyapunov import _chol, _solve_lyap_dense_check_args, _solve_lyap_lrcf_check_args
from pymor.algorithms.riccati import _solve_ricc_check_args, _solve_ricc_dense_check_args
from pymor.algorithms.to_matrix import to_matrix
from pymor.core.config import config
from pymor.core.defaults import defaults
from pymor.core.exceptions import InversionError
from pymor.operators.numpy import NumpyMatrixOperator


@defaults('bicgstab_tol', 'bicgstab_maxiter', 'spilu_drop_tol',
          'spilu_fill_factor', 'spilu_drop_rule', 'spilu_permc_spec', 'spsolve_permc_spec',
          'spsolve_keep_factorization', 'preconditioner_bandwidth',
          'lgmres_tol', 'lgmres_maxiter', 'lgmres_inner_m', 'lgmres_outer_k', 'least_squares_lsmr_damp',
          'least_squares_lsmr_atol', 'least_squares_lsmr_btol', 'least_squares_lsmr_conlim',
          'least_squares_lsmr_maxiter', 'least_squares_lsmr_show', 'least_squares_lsqr_atol',
          'least_squares_lsqr_btol', 'least_squares_lsqr_conlim', 'least_squares_lsqr_iter_lim',
          'least_squares_lsqr_show')
def solver_options(bicgstab_tol=1e-15,
                   bicgstab_maxiter=None,
                   spilu_drop_tol=1e-4,
                   spilu_fill_factor=10,
                   spilu_drop_rule=None,
                   spilu_permc_spec='COLAMD',
                   spsolve_permc_spec='COLAMD',
                   spsolve_keep_factorization=True,
                   preconditioner_bandwidth=None,
                   lgmres_tol=1e-5,
                   lgmres_maxiter=1000,
                   lgmres_inner_m=39,
                   lgmres_outer_k=3,
                   least_squares_lsmr_damp=0.0,
                   least_squares_lsmr_atol=1e-6,
                   least_squares_lsmr_btol=1e-6,
                   least_squares_lsmr_conlim=1e8,
                   least_squares_lsmr_maxiter=None,
                   least_squares_lsmr_show=False,
                   least_squares_lsqr_damp=0.0,
                   least_squares_lsqr_atol=1e-6,
                   least_squares_lsqr_btol=1e-6,
                   least_squares_lsqr_conlim=1e8,
                   least_squares_lsqr_iter_lim=None,
                   least_squares_lsqr_show=False):
    """Returns available solvers with default |solver_options| for the SciPy backend.

    Parameters
    ----------
    bicgstab_tol
        See :func:`scipy.sparse.linalg.bicgstab`.
    bicgstab_maxiter
        See :func:`scipy.sparse.linalg.bicgstab`.
    spilu_drop_tol
        See :func:`scipy.sparse.linalg.spilu`.
    spilu_fill_factor
        See :func:`scipy.sparse.linalg.spilu`.
    spilu_drop_rule
        See :func:`scipy.sparse.linalg.spilu`.
    spilu_permc_spec
        See :func:`scipy.sparse.linalg.spilu`.
    spsolve_permc_spec
        See :func:`scipy.sparse.linalg.spsolve`.
    spsolve_keep_factorization
        See :func:`scipy.sparse.linalg.spsolve`.
    lgmres_tol
        See :func:`scipy.sparse.linalg.lgmres`.
    lgmres_maxiter
        See :func:`scipy.sparse.linalg.lgmres`.
    lgmres_inner_m
        See :func:`scipy.sparse.linalg.lgmres`.
    lgmres_outer_k
        See :func:`scipy.sparse.linalg.lgmres`.
    least_squares_lsmr_damp
        See :func:`scipy.sparse.linalg.lsmr`.
    least_squares_lsmr_atol
        See :func:`scipy.sparse.linalg.lsmr`.
    least_squares_lsmr_btol
        See :func:`scipy.sparse.linalg.lsmr`.
    least_squares_lsmr_conlim
        See :func:`scipy.sparse.linalg.lsmr`.
    least_squares_lsmr_maxiter
        See :func:`scipy.sparse.linalg.lsmr`.
    least_squares_lsmr_show
        See :func:`scipy.sparse.linalg.lsmr`.
    least_squares_lsqr_damp
        See :func:`scipy.sparse.linalg.lsqr`.
    least_squares_lsqr_atol
        See :func:`scipy.sparse.linalg.lsqr`.
    least_squares_lsqr_btol
        See :func:`scipy.sparse.linalg.lsqr`.
    least_squares_lsqr_conlim
        See :func:`scipy.sparse.linalg.lsqr`.
    least_squares_lsqr_iter_lim
        See :func:`scipy.sparse.linalg.lsqr`.
    least_squares_lsqr_show
        See :func:`scipy.sparse.linalg.lsqr`.

    Returns
    -------
    A dict of available solvers with default |solver_options|.
    """
    opts = {'scipy_bicgstab_spilu':     {'type': 'scipy_bicgstab_spilu',
                                         'tol': bicgstab_tol,
                                         'maxiter': bicgstab_maxiter,
                                         'spilu_drop_tol': spilu_drop_tol,
                                         'spilu_fill_factor': spilu_fill_factor,
                                         'spilu_drop_rule': spilu_drop_rule,
                                         'spilu_permc_spec': spilu_permc_spec},
            'scipy_bicgstab':           {'type': 'scipy_bicgstab',
                                         'tol': bicgstab_tol,
                                         'maxiter': bicgstab_maxiter},
            'scipy_spsolve':            {'type': 'scipy_spsolve',
                                         'permc_spec': spsolve_permc_spec,
                                         'keep_factorization': spsolve_keep_factorization},
            'scipy_lgmres':             {'type': 'scipy_lgmres',
                                         'tol': lgmres_tol,
                                         'maxiter': lgmres_maxiter,
                                         'inner_m': lgmres_inner_m,
                                         'outer_k': lgmres_outer_k},
            'scipy_lgmres_spilu':       {'type': 'scipy_lgmres_spilu',
                                         'tol': lgmres_tol,
                                         'maxiter': lgmres_maxiter,
                                         'inner_m': lgmres_inner_m,
                                         'outer_k': lgmres_outer_k,
                                         'preconditioner_bandwidth': preconditioner_bandwidth,
                                         'spilu_drop_tol': spilu_drop_tol,
                                         'spilu_fill_factor': spilu_fill_factor,
                                         'spilu_drop_rule': spilu_drop_rule,
                                         'spilu_permc_spec': spilu_permc_spec},
            'scipy_least_squares_lsqr': {'type': 'scipy_least_squares_lsqr',
                                         'damp': least_squares_lsqr_damp,
                                         'atol': least_squares_lsqr_atol,
                                         'btol': least_squares_lsqr_btol,
                                         'conlim': least_squares_lsqr_conlim,
                                         'iter_lim': least_squares_lsqr_iter_lim,
                                         'show': least_squares_lsqr_show}}

    if config.HAVE_SCIPY_LSMR:
        opts['scipy_least_squares_lsmr'] = {'type': 'scipy_least_squares_lsmr',
                                            'damp': least_squares_lsmr_damp,
                                            'atol': least_squares_lsmr_atol,
                                            'btol': least_squares_lsmr_btol,
                                            'conlim': least_squares_lsmr_conlim,
                                            'maxiter': least_squares_lsmr_maxiter,
                                            'show': least_squares_lsmr_show}

    return opts

pymor.bindings.scipy.solver_options = solver_options


def restrict_bandwidth(matrix, width):
    """Convert a scipy sparse matrix to a restricted bandwidth matrix by setting all elements outside the bandwidth to zero."""
    result = csr_array(matrix.copy())
    i, j = result.nonzero()
    result.data[np.abs(i - j) > width] = 0.
    result.eliminate_zeros()
    return result


@defaults('check_finite', 'default_solver', 'default_least_squares_solver')
def apply_inverse(op, V, initial_guess=None, options=None, least_squares=False, check_finite=True,
                  default_solver='scipy_spsolve', default_least_squares_solver='scipy_least_squares_lsmr'):
    """Solve linear equation system.

    Applies the inverse of `op` to the vectors in `V` using SciPy.

    Parameters
    ----------
    op
        The linear, non-parametric |Operator| to invert.
    V
        |VectorArray| of right-hand sides for the equation system.
    initial_guess
        |VectorArray| with the same length as `V` containing initial guesses
        for the solution.  Some implementations of `apply_inverse` may
        ignore this parameter.  If `None` a solver-dependent default is used.
    options
        The |solver_options| to use (see :func:`solver_options`).
    least_squares
        If `True`, return least squares solution.
    check_finite
        Test if solution only contains finite values.
    default_solver
        Default solver to use (scipy_spsolve, scipy_bicgstab, scipy_bicgstab_spilu,
        scipy_lgmres, scipy_least_squares_lsmr, scipy_least_squares_lsqr).
    default_least_squares_solver
        Default solver to use for least squares problems (scipy_least_squares_lsmr,
        scipy_least_squares_lsqr).

    Returns
    -------
    |VectorArray| of the solution vectors.
    """
    assert V in op.range
    assert initial_guess is None or initial_guess in op.source and len(initial_guess) == len(V)

    if isinstance(op, NumpyMatrixOperator):
        matrix = op.matrix
    else:
        from pymor.algorithms.to_matrix import to_matrix
        matrix = to_matrix(op)

    options = _parse_options(options, solver_options(), default_solver, default_least_squares_solver, least_squares)

    V = V.to_numpy()
    initial_guess = initial_guess.to_numpy() if initial_guess is not None else None
    promoted_type = np.promote_types(matrix.dtype, V.dtype)
    R = np.empty((len(V), matrix.shape[1]), dtype=promoted_type)

    if options['type'] == 'scipy_bicgstab':
        for i, VV in enumerate(V):
            R[i], info = bicgstab(matrix, VV, initial_guess[i] if initial_guess is not None else None,
                                  tol=options['tol'], maxiter=options['maxiter'], atol='legacy')
            if info != 0:
                if info > 0:
                    raise InversionError(f'bicgstab failed to converge after {info} iterations')
                else:
                    raise InversionError(f'bicgstab failed with error code {info} (illegal input or breakdown)')
    elif options['type'] == 'scipy_bicgstab_spilu':
        ilu = spilu(matrix, drop_tol=options['spilu_drop_tol'], fill_factor=options['spilu_fill_factor'],
                    drop_rule=options['spilu_drop_rule'], permc_spec=options['spilu_permc_spec'])
        precond = LinearOperator(matrix.shape, ilu.solve)
        for i, VV in enumerate(V):
            R[i], info = bicgstab(matrix, VV, initial_guess[i] if initial_guess is not None else None,
                                  tol=options['tol'], maxiter=options['maxiter'], M=precond, atol='legacy')
            if info != 0:
                if info > 0:
                    raise InversionError(f'bicgstab failed to converge after {info} iterations')
                else:
                    raise InversionError(f'bicgstab failed with error code {info} (illegal input or breakdown)')
    elif options['type'] == 'scipy_spsolve':
        try:
            # maybe remove unusable factorization:
            if hasattr(matrix, 'factorization'):
                fdtype = matrix.factorizationdtype
                if not np.can_cast(V.dtype, fdtype, casting='safe'):
                    del matrix.factorization

            if hasattr(matrix, 'factorization'):
                # we may use a complex factorization of a real matrix to
                # apply it to a real vector. In that case, we downcast
                # the result here, removing the imaginary part,
                # which should be zero.
                R = matrix.factorization.solve(V.T).T.astype(promoted_type, copy=False)
            elif options['keep_factorization']:
                # the matrix is always converted to the promoted type.
                # if matrix.dtype == promoted_type, this is a no_op
                matrix.factorization = splu(matrix_astype_nocopy(matrix.tocsc(), promoted_type),
                                            permc_spec=options['permc_spec'])
                matrix.factorizationdtype = promoted_type
                R = matrix.factorization.solve(V.T).T
            else:
                # the matrix is always converted to the promoted type.
                # if matrix.dtype == promoted_type, this is a no_op
                R = spsolve(matrix_astype_nocopy(matrix, promoted_type), V.T, permc_spec=options['permc_spec']).T
        except RuntimeError as e:
            raise InversionError(e) from e
    elif options['type'] == 'scipy_lgmres':
        for i, VV in enumerate(V):
            R[i], info = lgmres(matrix, VV, initial_guess[i] if initial_guess is not None else None,
                                rtol=options['tol'],
                                atol=options['tol'],
                                maxiter=options['maxiter'],
                                inner_m=options['inner_m'],
                                outer_k=options['outer_k'])
            if info > 0:
                raise InversionError(f'lgmres failed to converge after {info} iterations')
            assert info == 0
    elif options['type'] == 'scipy_lgmres_spilu':
        p = restrict_bandwidth(matrix, options['preconditioner_bandwidth']) if options['preconditioner_bandwidth'] is not None else matrix
        ilu = spilu(p.tocsc(), drop_tol=options['spilu_drop_tol'], fill_factor=options['spilu_fill_factor'],
                    drop_rule=options['spilu_drop_rule'], permc_spec=options['spilu_permc_spec'])
        precond = LinearOperator(matrix.shape, ilu.solve)
        for i, VV in enumerate(V):
            R[i], info = lgmres(matrix, VV, initial_guess[i] if initial_guess is not None else None,
                                rtol=options['tol'],
                                atol=options['tol'],
                                maxiter=options['maxiter'],
                                inner_m=options['inner_m'],
                                outer_k=options['outer_k'],
                                M=precond)
            if info > 0:
                raise InversionError(f'lgmres failed to converge after {info} iterations')
            assert info == 0
    elif options['type'] == 'scipy_least_squares_lsmr':
        from scipy.sparse.linalg import lsmr
        for i, VV in enumerate(V):
            R[i], info, itn, _, _, _, _, _ = lsmr(matrix, VV,
                                                  damp=options['damp'],
                                                  atol=options['atol'],
                                                  btol=options['btol'],
                                                  conlim=options['conlim'],
                                                  maxiter=options['maxiter'],
                                                  show=options['show'],
                                                  x0=initial_guess[i] if initial_guess is not None else None)
            assert 0 <= info <= 7
            if info == 7:
                raise InversionError(f'lsmr failed to converge after {itn} iterations')
    elif options['type'] == 'scipy_least_squares_lsqr':
        for i, VV in enumerate(V):
            R[i], info, itn, _, _, _, _, _, _, _ = lsqr(matrix, VV,
                                                        damp=options['damp'],
                                                        atol=options['atol'],
                                                        btol=options['btol'],
                                                        conlim=options['conlim'],
                                                        iter_lim=options['iter_lim'],
                                                        show=options['show'],
                                                        x0=initial_guess[i] if initial_guess is not None else None)
            assert 0 <= info <= 7
            if info == 7:
                raise InversionError(f'lsmr failed to converge after {itn} iterations')
    else:
        raise ValueError('Unknown solver type')

    if check_finite:
        if not np.isfinite(np.sum(R)):
            raise InversionError('Result contains non-finite values')

    return op.source.from_numpy(R)

pymor.bindings.scipy.apply_inverse = apply_inverse

### Patch Operator to allow use of scipy_lgmres_spilu -


from pymor.operators.interface import Operator

from numbers import Number

import numpy as np

from pymor.algorithms import genericsolvers
from pymor.core.base import abstractmethod
from pymor.core.defaults import defaults
from pymor.core.exceptions import InversionError, LinAlgError
from pymor.parameters.base import ParametricObject
from pymor.parameters.functionals import ParameterFunctional
from pymor.vectorarrays.interface import VectorArray
from pymor.vectorarrays.numpy import NumpyVectorSpace


@patch
def apply_inverse(self:Operator, V, mu=None, initial_guess=None, least_squares=False):
    """Apply the inverse operator.

    Parameters
    ----------
    V
        |VectorArray| of vectors to which the inverse operator is applied.
    mu
        The |parameter values| for which to evaluate the inverse operator.
    initial_guess
        |VectorArray| with the same length as `V` containing initial guesses
        for the solution.  Some implementations of `apply_inverse` may
        ignore this parameter.  If `None` a solver-dependent default is used.
    least_squares
        If `True`, solve the least squares problem::

            u = argmin ||op(u) - v||_2.

        Since for an invertible operator the least squares solution agrees
        with the result of the application of the inverse operator,
        setting this option should, in general, have no effect on the result
        for those operators. However, note that when no appropriate
        |solver_options| are set for the operator, most implementations
        will choose a least squares solver by default which may be
        undesirable.

    Returns
    -------
    |VectorArray| of the inverse operator evaluations.

    Raises
    ------
    InversionError
        The operator could not be inverted.
    """
    assert V in self.range
    assert initial_guess is None or initial_guess in self.source and len(initial_guess) == len(V)
    from pymor.operators.constructions import FixedParameterOperator
    assembled_op = self.assemble(mu)
    if assembled_op != self and not isinstance(assembled_op, FixedParameterOperator):
        return assembled_op.apply_inverse(V, initial_guess=initial_guess, least_squares=least_squares)

    options = self.solver_options.get('inverse') if self.solver_options else None
    options = (None if options is None else
               {'type': options} if isinstance(options, str) else
               options.copy())
    solver_type = None if options is None else options['type']

    if self.linear:
        if solver_type is None or solver_type == 'to_matrix' or solver_type == 'scipy_lgmres_spilu':
            mat_op = None
            if not hasattr(self, '_mat_op'):
                if solver_type is None:
                    self.logger.warning(f'No specialized linear solver available for {self}.')
                    self.logger.warning('Trying to solve by converting to NumPy/SciPy matrix.')
                from pymor.algorithms.rules import NoMatchingRuleError
                try:
                    from pymor.algorithms.to_matrix import to_matrix
                    from pymor.operators.numpy import NumpyMatrixOperator
                    mat = to_matrix(assembled_op, mu=mu)
                    mat_op = NumpyMatrixOperator(mat, solver_options=self.solver_options)
                    if not self.parametric:
                        self._mat_op = mat_op
                except (NoMatchingRuleError, NotImplementedError) as e:
                    if solver_type == 'to_matrix':
                        raise InversionError from e
                    else:
                        self.logger.warning('Failed.')
            else:
                mat_op = self._mat_op
            if mat_op is not None:
                v = mat_op.range.from_numpy(V.to_numpy())
                i = None if initial_guess is None else mat_op.source.from_numpy(initial_guess.to_numpy())
                u = mat_op.apply_inverse(v, initial_guess=i, least_squares=least_squares)
                return self.source.from_numpy(u.to_numpy())
        self.logger.warning('Solving with unpreconditioned iterative solver.')
        return genericsolvers.apply_inverse(assembled_op, V, initial_guess=initial_guess,
                                            options=options, least_squares=least_squares)
    else:
        from pymor.algorithms.newton import newton
        from pymor.core.exceptions import NewtonError

        assert solver_type is None or solver_type == 'newton'
        options = options or {}
        options.pop('type', None)
        options['least_squares'] = least_squares

        with self.logger.block('Solving nonlinear problem using newton algorithm ...'):
            R = V.empty(reserve=len(V))
            for i in range(len(V)):
                try:
                    R.append(newton(self, V[i],
                                    initial_guess=initial_guess[i] if initial_guess is not None else None,
                                    mu=mu,
                                    **options)[0])
                except NewtonError as e:
                    raise InversionError(e) from e
        return R

### Fix get_defaults bug


import pymor.core.defaults
from pymor.core.defaults import _default_container


def get_defaults(user=True, file=True, code=True):
    """Get |default| values.

    Returns all |default| values as a dict. The parameters can be set to filter by type.

    Parameters
    ----------
    user
        If `True`, returned dict contains defaults that have been set by the user
        with :func:`set_defaults`.
    file
        If `True`, returned dict contains defaults that have been loaded from file.
    code
        If `True`, returned dict contains unmodified default values.
    """
    defaults = {}
    for k in _default_container.keys():
        v, t = _default_container.get(k)
        if t == 'user' and user:
            defaults[k] = v
        if t == 'file' and file:
            defaults[k] = v
        if t == 'code' and code:
            defaults[k] = v
    return defaults

pymor.core.defaults.get_defaults = get_defaults