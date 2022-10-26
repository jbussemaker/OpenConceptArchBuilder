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
import openmdao.api as om
import openconcept.api as oc
from openconcept.utilities.math.multiply_divide_comp import ElementMultiplyDivideComp
from openconcept.analysis.aerodynamics import PolarDrag
from openconcept.utilities.math.integrals import Integrator
from oc_architecting.architecture import *
from oc_architecting.arch_group import DynamicPropulsionArchitecture
from openmdao.api import  IndepVarComp

from examples.methods.weights_twin_hybrid import (
    WingWeight_SmallTurboprop,
    EmpennageWeight_SmallTurboprop,
    FuselageWeight_SmallTurboprop,
    NacelleWeight_MultiTurboprop,
    LandingGearWeight_SmallTurboprop,
    FuelSystemWeight_SmallTurboprop,
    EquipmentWeight_SmallTurboprop,
)

__all__ = ["DynamicACModel"]


class DynamicACModel(oc.IntegratorGroup):
    """
    OpenConcept-compliant aircraft model. Should be created using the DynamicACModel.factory function (see below).

    Note: the aircraft weight is calculated by simply subtracting the fuel flow from the MTOW over the course of the
    mission. The propulsion architecture system weight is added as an output, and should be integrated into the OEW
    external to OpenConcept!

    Usage in the setup function of your main analysis group:
    ```
    arch = PropSysArch(...)

    mission_model = MissionWithReserve(  # Can be replaced by any other OpenConcept mission class
        num_nodes=nn,
        aircraft_model=DynamicACModel.factory(arch),
    )
    ```

    Options
    -------
        num_nodes : float
            Number of analysis points to run (default 1)
        flight_phase : str|None
            Name of the flight phase (default: None)
        architecture: PropSysArch
            Propulsion system architecture description to use.

    Inputs
    --------
        fltcond|*: float
            Flight conditions during the mission segment (vec)
                fltcond|rho     Air density     (kg/m**3)
                fltcond|Utrue   True airspeed   (m/s)
                fltcond|CL      Trimmed CL      (-)
                fltcond|q       Dynamic pressure (Pa)
        throttle: float
            Throttle input to the engine, fraction from 0-1 (vec, -)
        propulsor_active: float (either 0 or 1)
            A flag to indicate on or off for the connected propulsor either 1 or 0 (vec, -)
        duration: float
            The amount of time to finish the segment in seconds (scalar, s)
        ac|*: float
            Aircraft design parameters (scalar)
                ac|aero|polar|CD0_cruise    CD0 in cruise       (-)
                ac|aero|polar|CD0_TO        CD0 in take-off     (-)
                ac|aero|polar|e             Oswald factor       (-)
                ac|aero|wing|S_ref          Wing reference area (m**2)
                ac|aero|wing|AR             Wing aspect ratio   (-)
                ac|weights|MTOW             Max take-off weight (kg) <-- used as initial wt during mission simulation

    Outputs
    ------
        drag: float
            Total drag of the aircraft (Vec, N)
        thrust: float
            Total thrust of the propulsion system (Vec, N)
        weight: float
            Total weight of the aircraft, calculated from MTOW and fuel flow (Vec, kg)
        seg_fuel_used: float
            Total fuel used in the mission segment (Scalar, kg)
        propulsion_system_weight : float
            The weight of the propulsion system (Scalar, kg)
    """

    @classmethod
    def factory(cls, architecture: PropSysArch):
        def _factory(num_nodes=1, flight_phase=None):
            return cls(num_nodes=num_nodes, flight_phase=flight_phase, architecture=architecture)

        return _factory

    def initialize(self):
        self.options.declare("num_nodes", default=1)
        self.options.declare("flight_phase", default=None)
        self.options.declare("architecture", types=PropSysArch, desc="The propulsion system architecture definition")

    def setup(self):
        nn = self.options["num_nodes"]
        self._add_propulsion_model(nn)
        self._add_drag_model(nn)
        self._add_weight_model(nn)

    def _add_propulsion_model(self, nn):
        controls = self.add_subsystem('controls', IndepVarComp(), promotes_outputs=['*'])
        controls.add_output('prop|rpm', val=np.ones((nn,)) * 2500, units='rpm')

        self.add_subsystem(
            "propmodel",
            DynamicPropulsionArchitecture(num_nodes=nn, architecture=self.options["architecture"],
                                          flight_phase=self.options['flight_phase']),
            promotes_inputs=["fltcond|*", "throttle", "propulsor_active", "duration", "prop|rpm"],
            promotes_outputs=["fuel_flow", "thrust", "propulsion_system_weight", "power_rating"],
        )

    def _add_drag_model(self, nn):
        # Determine CD0 source based on flight phase
        flight_phase = self.options["flight_phase"]
        if flight_phase not in ["v0v1", "v1v0", "v1vr", "rotate"]:
            cd0_source = "ac|aero|polar|CD0_cruise"
        else:
            cd0_source = "ac|aero|polar|CD0_TO"

        # Add drag model based on simple drag polar model
        self.add_subsystem(
            "drag",
            PolarDrag(num_nodes=nn),
            promotes_inputs=["fltcond|CL", "ac|geom|*", ("CD0", cd0_source), "fltcond|q", ("e", "ac|aero|polar|e")],
            promotes_outputs=["drag"],
        )

    def _add_weight_model(self, nn):
        # Integrate fuel flow
        fuel_int = self.add_subsystem(
            'fuel_int', Integrator(num_nodes=nn, method='simpson', diff_units='s', time_setup='duration'),
            promotes_inputs=['*'], promotes_outputs=['*'])
        fuel_int.add_integrand('fuel_used', rate_name='fuel_flow', val=1.0, units='kg')

        # Calculate weight by subtracting fuel used from MTOW
        # Note that fuel used is accumulated over all mission phases, therefore fuel_used here represents the total fuel
        # used since the first mission phase
        self.add_subsystem(
            'weight', oc.AddSubtractComp(output_name='weight', input_names=['ac|weights|MTOW', 'fuel_used'], units='kg',
                                         vec_size=(1, nn), scaling_factors=[1, -1]),
            promotes_inputs=['*'], promotes_outputs=['weight'])

        # Calculate total fuel used in this mission segment
        self.add_subsystem(
            'seg_fuel_used', om.ExecComp(['seg_fuel_used=sum(fuel_used)'],
                                         seg_fuel_used={'val': 1.0, 'units': 'kg'},
                                         fuel_used={'val': np.ones((nn,)), 'units': 'kg'},
        ), promotes_inputs=['*'], promotes_outputs=['*'])


# class MissionWrapper(om.Group):
#
#     def initialize(self):
#         self.options.declare('mission')
#
#     def setup(self):
#         mission = self.options['mission']
#         self.add_subsystem('mission', mission)
#         mission.setup()
#
#         inp_comp = ArchSubSystem.create_top_level(self, 'mission', ['climb', 'cruise', 'descent'], 'acmodel.propmodel')
#         self.set_order([inp_comp.name, 'mission'])


if __name__ == "__main__":
    from oad_oc_link.missions.mission_profiles import MissionWithReserve
    from openconcept.analysis.performance.mission_profiles import FullMissionAnalysis

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
    prob.model = grp = om.Group()
    arch.create_top_level(
        grp, ["v0v1", "v1vr", "rotate", "v1v0", "engineoutclimb", "climb", "cruise", "descent"], "propmodel"
    )
    grp.add_subsystem(
        "analysis",
        FullMissionAnalysis(
            num_nodes=11,
            aircraft_model=DynamicACModel.factory(arch),
        ),
        promotes_inputs=["*"],
        promotes_outputs=["*"],
    )

    grp.add_design_var("ac|propulsion|mech_engine|rating", lower=50, upper=2000)
    grp.add_design_var("ac|propulsion|motor|rating", lower=50, upper=2000)
    grp.add_design_var("ac|weights|W_battery", lower=100, upper=3000)
    # grp.add_design_var('ac|propulsion|elec_engine|rating', lower=50, upper=2000)
    grp.add_design_var("ac|propulsion|mech_splitter|mech_DoH", lower=0.01, upper=0.99)  # used for parallel hybrid
    # grp.add_design_var('ac|propulsion|elec_splitter|elec_DoH', lower=0.01, upper=0.99)  # used for series hybrid

    prob.setup()
    om.n2(prob, show_browser=True)
