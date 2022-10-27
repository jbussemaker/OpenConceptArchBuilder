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

from oc_architecting.utils import *
from oc_architecting.architecture import *

__all__ = ["DynamicPropulsionArchitecture"]


class DynamicPropulsionArchitecture(om.Group):
    """
    A propulsion system architecture analysis group built-up from a PropSysArch definition.

    Propulsion system inputs (i.e. design variables and configuration parameters) are defined statically from
    ArchElement definitions. These parameters are normally optimized by the architecture optimizer, and therefore do not
    need to be known to the user of this analysis group.

    Options
    -------
        num_nodes : float
            Number of analysis points to run (default 1)

    Inputs
    --------
        prop|rpm: float
            propeller rpm (vec, RPM)
        fltcond|rho: float
            air density at the specific flight condition (vec, kg/m**3)
        fltcond|Utrue: float
            true airspeed at the flight condition (vec, m/s)
        throttle: float
            throttle input to the engine, fraction from 0-1 (vec, '')
        propulsor_active: float (either 0 or 1)
            a flag to indicate on or off for the connected propulsor either 1 or 0 (vec, '')
        duration: float
            the amount of time to finish the segment in seconds (Scalar, 's')

    Outputs
    ------
        propulsion_system_weight : float
            The weight of the propulsion system (Scalar, 'kg') Note: battery weight is included
        fuel_flow: float
            The fuel flow consumed in the segment (Vec, 'kg/s')
        soc: float
            State-of-charge along the segment (Vec, dimensionless)
        thrust: float
            The total thrust of the propulsion system (Vec, 'N')
    """

    def initialize(self):
        self.options.declare("num_nodes", default=1, desc="Number of mission analysis points to run")
        self.options.declare("flight_phase", default=None)
        self.options.declare("architecture", types=PropSysArch, desc="The propulsion system architecture definition")

    def setup(self):
        nn = self.options["num_nodes"]
        phase = self.options['flight_phase']
        arch: PropSysArch = self.options["architecture"]

        # Define inputs
        main_prop = arch.thrust.propellers[0]
        default_rpm = 2000.0 if main_prop is None else main_prop.default_rpm
        input_comp, input_map = collect_inputs(
            self,
            [
                (RPM_INPUT, "rpm", np.tile(default_rpm, nn)),
                (THROTTLE_INPUT, None, np.tile(1.0, nn)),
                (DURATION_INPUT, "s", 1.0),
                (ACTIVE_INPUT, None, np.tile(1.0, nn)),
                (FLTCOND_RHO_INPUT, "kg/m**3", np.tile(1.225, nn)),
                (FLTCOND_TAS_INPUT, "m/s", np.tile(100.0, nn)),
                (WEIGHT_INPUT, "kg", 1.0),
            ],
            name="propmodel_in_collect",
        )
        order = [input_comp.name]

        subsys_groups = []
        weight_outputs = []
        thrust_outputs = []
        fuel_flow_outputs = []
        soc_outputs = []
        power_outputs = []

        # Create thrust generation groups: propellers + gearboxes
        thrust_groups = arch.thrust.create_thrust_groups(self, nn)
        subsys_groups += thrust_groups.copy()

        weight_outputs += [grp.name + "." + WEIGHT_OUTPUT for grp in thrust_groups]
        thrust_outputs += [grp.name + "." + THRUST_OUTPUT for grp in thrust_groups]

        # Create mechanical power generation groups: motors or engines connected to the propellers
        mech_group, electric_power_gen_needed = arch.mech.create_mech_group(self, thrust_groups, nn, phase)
        subsys_groups += [mech_group]

        fuel_flow_outputs += [mech_group.name + "." + FUEL_FLOW_OUTPUT]
        power_outputs += [mech_group.name + "." + POWER_RATING_OUTPUT]
        weight_outputs += [mech_group.name + "." + WEIGHT_OUTPUT]

        order += [mech_group.name]
        order += [grp.name for grp in thrust_groups]

        # If needed, create electrical power generation groups: batteries, engines, etc.
        if electric_power_gen_needed:
            if arch.electric is None:
                raise RuntimeError("Electrical power generation is needed but no `ElectricPowerElements` is defined!")

            elec_group = arch.electric.create_electric_group(self, mech_group, thrust_groups, nn, phase)
            subsys_groups += [elec_group]

            fuel_flow_outputs += [elec_group.name + "." + FUEL_FLOW_OUTPUT]
            weight_outputs += [elec_group.name + "." + WEIGHT_OUTPUT]
            soc_outputs += [elec_group.name + "." + SOC_OUTPUT]

            order += [elec_group.name]

        # Connect inputs
        def _connect_input(input_name: str, groups: List[om.Group], group_input_name: str = None):
            for group in groups:
                self.connect(input_map[input_name], group.name + "." + (group_input_name or input_name))

        _connect_input(RPM_INPUT, thrust_groups)
        _connect_input(THROTTLE_INPUT, [mech_group])
        _connect_input(ACTIVE_INPUT, [mech_group])
        _connect_input(WEIGHT_INPUT, [mech_group])
        if arch.electric is not None:
            _connect_input(WEIGHT_INPUT, [elec_group])

        _connect_input(DURATION_INPUT, subsys_groups)
        _connect_input(FLTCOND_RHO_INPUT, subsys_groups)
        _connect_input(FLTCOND_TAS_INPUT, subsys_groups)

        # Create summed outputs
        ff_comp = create_output_sum(self, "fuel_flow", fuel_flow_outputs, "kg/s", n=nn)
        wt_comp = create_output_sum(self, "propulsion_system_weight", weight_outputs, "kg")
        th_comp = create_output_sum(self, "thrust", thrust_outputs, "N", n=nn)
        soc_comp = create_output_sum(self, "SOC", soc_outputs, n=nn)
        p_comp = create_output_sum(self, "power_rating", power_outputs, "kW")

        order += [ff_comp.name, wt_comp.name, th_comp.name, soc_comp.name,p_comp.name]

        # Define order to reduce feedback connections
        self.set_order(order)

        # self.nonlinear_solver = om.NewtonSolver(iprint=2, err_on_non_converge=False)
        # self.nonlinear_solver.linesearch = om.BoundsEnforceLS(print_bound_enforce=True)
        # self.options["assembled_jac_type"] = "csc"
        # self.linear_solver = om.DirectSolver(assemble_jac=True)
        # self.nonlinear_solver.options["solve_subsystems"] = False
        # self.nonlinear_solver.options["maxiter"] = 15
        # self.nonlinear_solver.options["atol"] = 1e-4
        # self.nonlinear_solver.options["rtol"] = 1e-4


if __name__ == "__main__":
    # arch = PropSysArch(  # Conventional without gearbox
    #     thrust=ThrustGenElements(propellers=[Propeller('prop1'), Propeller('prop2')]),
    #     mech=MechPowerElements(engines=[Engine('turboshaft'), Engine('turboshaft')]),
    # )
    # #
    # arch = PropSysArch(  # Conventional with gearbox
    #     thrust=ThrustGenElements(propellers=[Propeller('prop1'), Propeller('prop2')],
    #                              gearboxes=[Gearbox('gearbox1'), Gearbox('gearbox2')]),
    #     mech=MechPowerElements(engines=[Engine('turboshaft'), Engine('turboshaft')]),
    # )
    # #
    # arch = PropSysArch(  # All-electric with inverters
    #     thrust=ThrustGenElements(propellers=[Propeller('prop1'), Propeller('prop2')],
    #                              gearboxes=[Gearbox('gearbox1'), Gearbox('gearbox2')]),
    #     mech=MechPowerElements(motors=[Motor('elec_motor'), Motor('elec_motor')],
    #                            inverters=Inverter('inverter')),
    #     electric=ElectricPowerElements(dc_bus=DCBus('elec_bus'),
    #                                    batteries=Batteries('bat_pack')),
    # )
    #
    # arch = PropSysArch(  # series hybrid with one engine, battery and two motors
    #     thrust=ThrustGenElements(propellers=[Propeller('prop1'), Propeller('prop2')],
    #                              gearboxes=[Gearbox('gearbox1'), Gearbox('gearbox2')]),
    #     mech=MechPowerElements(motors=[Motor('elec_motor'), Motor('elec_motor')],
    #                            inverters=Inverter('inverter')),
    #     electric=ElectricPowerElements(dc_bus=DCBus('elec_bus'),
    #                                    splitter=ElecSplitter('splitter'),
    #                                    batteries=Batteries('bat_pack'),
    #                                    engines_dc=(Engine(name='turboshaft'), Generator(name='generator'),
    #                                                Rectifier(name='rectifier'))),
    # )
    #
    # arch = PropSysArch(  # turboelectric with one engine and two motors
    #     thrust=ThrustGenElements(propellers=[Propeller('prop1'), Propeller('prop2')],
    #                              gearboxes=[Gearbox('gearbox1'), Gearbox('gearbox2')]),
    #     mech=MechPowerElements(motors=[Motor('elec_motor'), Motor('elec_motor')],
    #                            inverters=Inverter('inverter')),
    #     electric=ElectricPowerElements(dc_bus=DCBus('elec_bus'),
    #                                    engines_dc=(Engine(name='turboshaft'), Generator(name='generator'),
    #                                                Rectifier(name='rectifier'))),
    # )
    #
    arch = PropSysArch(  # parallel hybrid system
        thrust=ThrustGenElements(
            propellers=[Propeller("prop1"), Propeller("prop2")], gearboxes=[Gearbox("gearbox1"), Gearbox("gearbox2")]
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
