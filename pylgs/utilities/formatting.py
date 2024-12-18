"""Formatting functions"""

# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/api/utilities/formatting.ipynb.

# %% auto 0
__all__ = ['si_prefixes', 'significant', 'significant_digits', 'prefix_format']

# %% ../../nbs/api/utilities/formatting.ipynb
import math
from numbers import Integral

# %% ../../nbs/api/utilities/formatting.ipynb
si_prefixes = {
    -30 : 'q',
    -27 : 'r',
    -24 : 'y',
    -21 : 'z',
    -18 : 'a',
    -15 : 'f',
    -12 : 'p',
    -9  : 'n',
    -6  : 'µ',
    -3  : 'm',
    0   : '',
    3   : 'k',
    6   : 'M',
    9   : 'G',
    12  : 'T',
    15  : 'P',
    18  : 'E',
    21  : 'Z',
    24  : 'Y',
    27  : 'R',
    30  : 'Q',
}

# %% ../../nbs/api/utilities/formatting.ipynb
def significant(x, n=1):
    if x == 0: return x
    return round(x, -int(math.floor(math.log10(abs(x)))) + (n - 1))

# %% ../../nbs/api/utilities/formatting.ipynb
def significant_digits(x, n=1):
    if x == 0: return x
    x = significant(x, n)
    return int(round(x * 10**(-int(math.floor(math.log10(abs(x)))) + (n - 1))))

# %% ../../nbs/api/utilities/formatting.ipynb
def prefix_format(x, precision=3):
    if x == 0: return f'{x}'
    e = (math.floor(math.log(abs(x), 10) + 2) // 3) * 3
    m = x/10**e
    if isinstance(precision, Integral):
        p = max(precision - math.ceil(math.log(abs(m), 10)), 0)
        return f'{m:.{p}f}{si_prefixes[e]}'
    else:
        precision = significant(precision/10**e)
        p = -math.floor(math.log(precision, 10))
        if p < 0: return f'({m:.1f}±{precision:.0f}){si_prefixes[e]}'
        return f'{m:.{p}f}({significant_digits(precision)}){si_prefixes[e]}'
