"""
The MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Copyright: (c) 2022, Deutsches Zentrum fuer Luft- und Raumfahrt e.V.
Contact: jasper.bussemaker@dlr.de
         mahmoud.fouda@dlr.de
"""

import numpy as np
from typing import *
import openmdao.api as om
from dataclasses import dataclass
from oc_architecting.defs import *
from oc_architecting.utils import *
from oc_architecting.architecture import *

from openconcept.components import SimplePropeller
from oc_architecting.components import SimpleGearbox

__all__ = [
    "ThrustGenElements",
    "Propeller",
    "Gearbox",
    "SHAFT_POWER_INPUT",
    "SHAFT_SPEED_INPUT",
    "RATED_POWER_INPUT",
    "THRUST_OUTPUT",
    "RPM_INPUT",
]

RPM_INPUT = "prop|rpm"  # control input passing from mission analysis
SHAFT_POWER_INPUT = "shaft_power"  # shaft power input passing from mech group to gearbox or propeller
SHAFT_SPEED_INPUT = "shaft_speed"  # shaft speed input passing from mech group to gearbox or propeller
RATED_POWER_INPUT = "rated_power"  # rated power input passing from mech group to gearbox or propeller

THRUST_OUTPUT = "thrust"


@dataclass(frozen=False)
class Propeller(ArchElement):
    """Represents a propeller for thrust generation."""

    name: str = "prop"

    blades: int = 4
    diameter: float = 2.5  # m
    # power_rating: float = 240  # kW , passed from mech group

    design_cp: float = 0.2  # Cruise power coefficient
    design_adv_ratio: float = 2.2  # V/n/D (advance ratio)

    default_rpm: float = 2000.0  # Default RPM if not overriden by design variables


@dataclass(frozen=False)
class Gearbox(ArchElement):
    """Mechanical reduction gearbox for the propeller. Output RPM is set by the propeller RPM."""

    name: str = "gearbox"

    # input_rpm: float = 5500  # rpm, typical shaft speed input either from engine, motor, or mech bus
    # output_rpm: float = 2000  # rpm, typical shaft speed out to propulsive device, coming from mission group
    # power_rating: float = 240  # kW, passed from engine, motor, or engine+motor


@dataclass(frozen=False)
class ThrustGenElements(ArchSubSystem):
    """Thrust generation elements in the propulsion system architecture."""

    propellers: List[Propeller]
    gearboxes: Optional[List[Optional[Gearbox]]] = None

    def get_dv_defs(self) -> List[Tuple[str, List[str], str, Any]]:
        thrust_dvs = []
        if self.propellers is not None:
            diameter_paths = ["thrust%d.diameter" % (i + 1,) for i in range(len(self.propellers))]
            thrust_dvs += (("ac|propulsion|propeller|diameter", diameter_paths, "m", self.propellers[0].diameter),)

        return thrust_dvs

    def create_thrust_groups(self, arch: om.Group, nn: int) -> List[om.Group]:
        """
        Creates thrust generation groups for each propeller.

        Inputs: RPM_INPUT[nn], DURATION_INPUT, FLTCOND_RHO_INPUT[nn], FLTCOND_TAS_INPUT[nn], SHAFT_POWER_INPUT[nn]
        Outputs: THRUST_OUTPUT[nn], WEIGHT_OUTPUT
        """

        # Get propeller and gearbox definitions
        props = self.propellers
        gearboxes = self.gearboxes
        if gearboxes is not None:
            if len(gearboxes) != len(props):
                raise RuntimeError("Nr of props (%d) not the same as gearboxes (%d)" % (len(props), len(gearboxes)))
        else:
            gearboxes = [None for _ in range(len(props))]

        # Create thrust groups
        groups = []
        for i, prop in enumerate(props):
            thrust_group: om.Group = arch.add_subsystem("thrust%d" % (i + 1,), om.Group())
            groups.append(thrust_group)

            # Define inputs
            input_comp, input_map = collect_inputs(
                thrust_group,
                [
                    (RPM_INPUT, "rpm", np.tile(prop.default_rpm, nn)),
                    (DURATION_INPUT, "s", 1.0),
                    (SHAFT_POWER_INPUT, "kW", np.tile(1.0, nn)),
                    (SHAFT_SPEED_INPUT, "rpm", 1.0),  # needed for gearbox component
                    (RATED_POWER_INPUT, "kW", 1.0),  # needed to size gearbox and propeller
                    # propeller inputs
                    ("diameter", "m", prop.diameter),
                    # ('power', 'kW', prop.power_rating), passed from mech group
                    # gearbox inputs: there are no inputs
                ],
                name="thrust%d" % (i + 1,) + "_in_collect",
            )
            order = [input_comp.name]

            prop.blades=int(prop.blades)
            # Create propeller
            prop_sys = thrust_group.add_subsystem(
                prop.name,
                SimplePropeller(
                    num_nodes=nn, num_blades=prop.blades, design_J=prop.design_adv_ratio, design_cp=prop.design_cp
                ),
                promotes_inputs=[FLTCOND_RHO_INPUT, FLTCOND_TAS_INPUT],
                promotes_outputs=[THRUST_OUTPUT],
            )
            shaft_power_input_param = prop_shaft_power_in = prop_sys.name + ".shaft_power_in"
            weights = [prop_sys.name + ".component_weight"]

            thrust_group.connect(input_map[RPM_INPUT], prop_sys.name + ".rpm")
            thrust_group.connect(input_map["diameter"], prop_sys.name + ".diameter")
            # thrust_group.connect(input_map['power'], prop_sys.name+'.power_rating')
            thrust_group.connect(input_map[RATED_POWER_INPUT], prop_sys.name + ".power_rating")

            # Create optional gearbox
            gearbox = gearboxes[i]
            if gearbox is not None:
                gear_sys = thrust_group.add_subsystem(gearbox.name, SimpleGearbox(num_nodes=nn))
                shaft_power_input_param = gear_sys.name + ".shaft_power_in"
                weights += [gear_sys.name + ".component_weight"]

                # Connect gearbox to propeller
                thrust_group.connect(gear_sys.name + ".shaft_power_out", prop_shaft_power_in)

                # define shaft speed input to gearbox
                thrust_group.connect(input_map[SHAFT_SPEED_INPUT], gear_sys.name + ".shaft_speed_in")
                # get rpm as scalar
                scalify_rpm = thrust_group.add_subsystem(
                    "scalify_rpm",
                    ScalifyComponent(
                        vars=[
                            (input_map[RPM_INPUT], RPM_INPUT + "_scalar", nn, "rpm"),
                        ]
                    ),
                    promotes_inputs=["*"],
                    promotes_outputs=["*"],
                )
                # define shaft speed output of gearbox
                thrust_group.connect(RPM_INPUT + "_scalar", gear_sys.name + ".shaft_speed_out")
                order += [scalify_rpm.name]

                # define power rating of gearbox
                thrust_group.connect(input_map[RATED_POWER_INPUT], gear_sys.name + ".shaft_power_rating")
                # add gearbox to group order
                order += [gear_sys.name]

            order += [prop_sys.name]

            # Define shaft power input parameter
            thrust_group.connect(input_map[SHAFT_POWER_INPUT], shaft_power_input_param)

            # Sum weight outputs
            wt_comp = create_output_sum(thrust_group, WEIGHT_OUTPUT, weights, "kg")
            order += [wt_comp.name]

            thrust_group.set_order(order)

        return groups
