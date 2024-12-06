# pyLGS


<!-- WARNING: THIS FILE WAS AUTOGENERATED! DO NOT EDIT! -->

pyLGS performs simulations of the atomic physics of cw, modulated, and
pulsed laser guide stars. The effects of the full atomic structure,
atomic velocity distribution, one or multiple pump fields, the
geomagnetic field, velocity-changing and spin-randomizing collisions,
and atomic recoil are all taken into account.

## Installation

pyLGS uses the CVODE library from the [SUNDIALS
package](https://computing.llnl.gov/projects/sundials), with
[scikits.odes](https://scikits-odes.readthedocs.io/en/latest/installation.html)
as the Python interface. Before installing pyLGS you may need to install
SUNDIALS, as described in the
[scikits.odes](https://scikits-odes.readthedocs.io/en/latest/installation.html)
and [SUNDIALS](https://computing.llnl.gov/projects/sundials)
documentation. pyLGS is tested with SUNDIALS version 6.5.1.

``` sh
pip install pylgs
```

## How to use

Import the package:

``` python
from pylgs.lgssystem import LGSSystem
```

List available atomic systems for an LGS model:

``` python
LGSSystem.names()
```

    ['NaD1', 'Na330', 'NaD2', 'NaD2_Repump', 'NaD1_Toy']

Choose an atomic system and set values for parameters that will not be
varied:

``` python
lgs = LGSSystem(
    'NaD2_Repump', 
    {
        'EllipticityDegrees1': 45.,
        'PolarizationAngleDegrees1': 0,
        'DetuningHz1': 1.0832e9,
        'LaserWidthHz1': 10.0e6,
        'EllipticityDegrees2': 45.,
        'PolarizationAngleDegrees2': 0,
        'DetuningHz2': -6.268e8 + 1.e8,
        'LaserWidthHz2': 10.0e6,
        'MagneticZenithDegrees': 45.,
        'MagneticAzimuthDegrees': 45.,
        'SDampingCollisionRatePerS': 4081.63,
        'BeamTransitRatePerS': 131.944,
        'VccRatePerS': 28571.,
        'TemperatureK': 185.,
        'RecoilParameter': 1
    }
)
```

Define sample values for the varying parameters:

``` python
params = {'IntensitySI1': 5., 'IntensitySI2': 46., 'BFieldG': 0.5}
```

Build a steady-state model with adaptively refined velocity groups based
on the sample parameters:

``` python
model = lgs.adaptive_stationary_model(params)
```

Solve the model for the steady state using the sample parameters:

``` python
sol = model.solve(params)
```

Find the total return flux:

``` python
model.total_flux(sol).item()
```

    7709.051827675756

Plot the return flux as a function of atomic velocity:

``` python
model.flux_distribution(sol).visualize()
```

![](index_files/figure-commonmark/cell-9-output-1.png)

Plot the ground and excited state populations as a function of atomic
velocity:

``` python
model.level_population_distribution(sol).visualize()
```

![](index_files/figure-commonmark/cell-10-output-1.png)

Plot the real and imaginary parts of all density-matrix elements:

``` python
model.velocity_normalize(sol).visualize(line_width=1)
```

![](index_files/figure-commonmark/cell-11-output-1.png)
