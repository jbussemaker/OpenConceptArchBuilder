# OpenConcept Architecture Builder

The architecture builder enables the easy definition of propulsion system architectures for use in analyses of
[OpenConcept](https://github.com/mdolab/openconcept).

## Installation

```pip install -r requirements.txt```

## Usage

```python
import openmdao.api as om
from oc_architecting.arch_group import *
from oc_architecting.architecture import *

arch = PropSysArch(  # parallel hybrid system
    thrust=ThrustGenElements(
        propellers=[Propeller("prop1"), Propeller("prop2")],
        gearboxes=[Gearbox("gearbox1"), Gearbox("gearbox2")]
    ),
    mech=MechPowerElements(
        engines=[Engine("turboshaft"), Engine("turboshaft")],
        motors=[Motor("motor"), Motor("motor")],
        mech_buses=MechBus("mech_bus"),
        mech_splitters=MechSplitter("mech_splitter"),
        inverters=Inverter("inverter"),
    ),
    electric=ElectricPowerElements(dc_bus=DCBus("elec_bus"), batteries=Batteries("bat_pack")),
)

prob = om.Problem()
prob.model = DynamicPropulsionArchitecture(num_nodes=11, architecture=arch)
prob.setup()
om.n2(prob, show_browser=True)
```

## Citation

Please cite this software by reference to the [conference paper](https://www.researchgate.net/publication/363405270_AUTOMATED_HYBRID_PROPULSION_MODEL_CONSTRUCTION_FOR_CONCEPTUAL_AIRCRAFT_DESIGN_AND_OPTIMIZATION):

Mahmoud Fouda, Eytan J. Adler, et al., "Automated Hybrid Propulsion Model Construction for Conceptual Aircraft Design and Optimization", 33rd Congress of the International Council of the Aeronautical Sciences (ICAS), Stockholm, Sweden, September 2022.

```
@InProceedings{Fouda2022,
  author    = {Mahmoud Fouda and Eytan J. Adler and Jasper H. Bussemaker and Joaquim R.R.A. Martins and D.F. Kurtulus and Luca Boggero and Björn Nagel},
  booktitle = {33rd Congress of the International Council of the Aeronautical Sciences, ICAS 2022},
  title     = {Automated Hybrid Propulsion Model Construction for Conceptual Aircraft Design and Optimization},
  year      = {2022},
  address   = {Stockholm, Sweden},
  month     = {September},
}
```
