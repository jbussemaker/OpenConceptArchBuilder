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
from oc_architecting.elements.thrust import *

from openconcept.components import SimpleTurboshaft, SimpleMotor, PowerSplit
from oc_architecting.components import SimpleConverterInverted, SimpleMechBus
from openconcept.utilities.math import AddSubtractComp, ElementMultiplyDivideComp
from openmdao.api import ExplicitComponent


__all__ = [
    "MechPowerElements",
    "Engine",
    "Motor",
    "Inverter",
    "FUEL_FLOW_OUTPUT",
    "ELECTRIC_POWER_OUTPUT",
    "MechSplitter",
    "MechBus",
    "THROTTLE_INPUT",
    "ACTIVE_INPUT",
    "POWER_RATING_OUTPUT",
    "WEIGHT_INPUT",
]

THROTTLE_INPUT = "throttle"
ACTIVE_INPUT = "propulsor_active"

FUEL_FLOW_OUTPUT = "fuel_flow"
ELECTRIC_POWER_OUTPUT = "motors_elec_power"
POWER_RATING_OUTPUT = "power_rating"
WEIGHT_INPUT = 'ac|weights|MTOW'




@dataclass(frozen=False)
class Engine(ArchElement):
    """Conventional turboshaft engine."""

    name: str = "turboshaft"

    power_rating: float = 260.0  # kW

    specific_weight: float = 0.14 / 1000  # kg/kW
    base_weight: float = 104  # kg
    psfc: float = 0.6  # kg/W/s
    output_rpm: float = 6000  # rpm


@dataclass(frozen=False)
class Motor(ArchElement):
    """Electric motor."""

    name: str = "motor"

    power_rating: float = 260.0  # kW
    efficiency: float = 0.97
    specific_weight: float = 1.0 / 5000  # kg/kW
    base_weight: float = 0.0  # kg
    cost_inc: float = 100.0 / 745.0  # $ per watt
    cost_base: float = 1.0  # $ per base
    output_rpm: float = 6000  # rpm


@dataclass(frozen=False)
class Inverter(ArchElement):
    """A DC to AC inverter."""

    name: str = "inverter"

    efficiency: float = 0.97
    # power_rating: float = 260.  # kW, passed from electric motor
    specific_weight: float = 1.0 / (10 * 1000)  # kg/kW
    base_weight: float = 0.0  # kg
    cost_inc: float = 100.0 / 745.0  # $ per watt
    cost_base: float = 1.0  # $ per base


@dataclass(frozen=False)
class MechSplitter(ArchElement):
    """mech power splitter to divide a power input to two outputs A and B based on a split fraction and
    efficiency loss"""

    name: str = "mech_splitter"

    power_rating: float = 99999999  # 'W', maximum power rating of split component
    efficiency: float = 1.0  # efficiency defines the loss of combining eng+motor shaft power
    split_rule: str = "fraction"  # this sets the rule to always use a fraction between 0 and 1
    mech_DoH: float = 0.5  # degree of hybridization between eng & motor for delivering shaft power, 0 =< mech_DoH =< 1


@dataclass(frozen=False)
class MechBus(ArchElement):

    name: str = "mech_bus"

    efficiency: float = 0.95  # efficiency loss to combine eng and motor shaft powers
    rpm_out: float = 6000  # output rpm of the mechanical bus to be connected to gearbox


@dataclass(frozen=False)
class MechPowerElements(ArchSubSystem):
    """Mechanical power generation elements in the propulsion system architecture. It is assumed that mechanical power
    is generated near the propellers, and therefore the local sub-architecture of the mechanical elements is replicated
    for each propeller."""

    # specify a list of elements to distribute over the propellers
    # for inverters, mech_bus and splitter, it is possible to specify one element to be replicated for each propeller,
    engines: Optional[List[Optional[Engine]]] = None  # must be a list of engines
    motors: Optional[List[Optional[Motor]]] = None  # must be a list of motors
    inverters: Optional[Union[Inverter, List[Optional[Inverter]]]] = None
    mech_buses: Optional[Union[MechBus, List[Optional[MechBus]]]] = None
    mech_splitters: Optional[Union[MechSplitter, List[Optional[MechSplitter]]]] = None

    mech_doh: Optional[Dict[str, float]] = None

    def get_dv_defs(self) -> List[Tuple[str, List[str], str, Any]]:
        if all(v is None for v in self.motors):
            self.motors = None
        if all(v is None for v in self.engines):
            self.engines = None
        mech_dvs = []
        eng_rating_paths = []
        motor_rating_paths = []

        # Engine power design variable is only activated when there is an engine
        if self.engines is not None and type(self.engines) == list:
            for i in range(len(self.engines)):
                if self.engines[i] is not None:
                    eng_rating_paths.append("mech.mech%d.eng_rating" % (i + 1,))
                    first_engine_position=i
            mech_dvs += (("ac|propulsion|mech_engine|rating", eng_rating_paths, "kW/kg", self.engines[first_engine_position].power_rating),)

        # Motor power design variable is only activated when there is a motor
        if self.motors is not None and type(self.motors) == list:
            for i in range(len(self.motors)):
                if self.motors[i] is not None:
                    motor_rating_paths.append("mech.mech%d.motor_rating" % (i + 1,))
                    first_motor_position=i
            mech_dvs += (("ac|propulsion|motor|rating", motor_rating_paths, "kW/kg", self.motors[first_motor_position].power_rating),)

        # if self.engines is not None and self.motors is not None:
        #     if self.mech_splitters is not None:
        #         mech_doh_paths = ['mech.mech%d.mech_DoH' % (i + 1,) for i in range(len(self.engines))]
        #         mech_dvs += ('ac|propulsion|mech_splitter|mech_DoH', mech_doh_paths, None,
        #                      self.mech_splitters.mech_DoH),

        return mech_dvs

    def create_mech_group(self, arch: om.Group, thrust_groups: List[om.Group], nn: int, phase: str) -> Tuple[om.Group, bool]:
        """
        Creates the mechanical power group and returns whether electric power generation is needed.

        Inputs: THROTTLE_INPUT[nn], DURATION_INPUT, FLTCOND_RHO_INPUT[nn], FLTCOND_TAS_INPUT[nn], ACTIVE_INPUT
        Outputs: FUEL_FLOW_OUTPUT[nn], ELECTRIC_LOAD_OUTPUT[nn], WEIGHT_OUTPUT
        """

        # Prepare components
        n_thrust = len(thrust_groups)
        engines = self.engines
        motors = self.motors
        inverters = self.inverters
        mech_buses = self.mech_buses
        mech_splitters = self.mech_splitters
        mech_doh = (self.mech_doh or {}).get(phase, 1.)

        # check engines and motor inputs are passed as lists with n_thrust members
        if engines is not None:
            if type(engines) != list:
                raise ValueError("engines must be a list of engines")
            else:
                if len(engines) != n_thrust:
                    raise ValueError("list of engines must be of the same length as list of propellers")
                else:
                    if len(engines) == 1:
                        engines = engines[0]
                    elif len(engines) < 1:
                        raise ValueError("engines list cannot be empty")

        if motors is not None:
            if type(motors) != list:
                raise ValueError("motors must be a list of motors")
            else:
                if len(motors) != n_thrust:
                    raise ValueError("list of motors must have the same length as list of propellers")
                else:
                    if len(motors) == 1:
                        motors = motors[0]
                    elif len(motors) < 1:
                        raise ValueError("motors list cannot be empty")

        # prepare inputs for processing
        if engines is None or isinstance(engines, Engine):
            engines = [engines for _ in range(n_thrust)]
        if motors is None or isinstance(motors, Motor):
            motors = [motors for _ in range(n_thrust)]
        if inverters is None or isinstance(inverters, Inverter):
            inverters = [inverters for _ in range(n_thrust)]
        if mech_buses is None or isinstance(mech_buses, MechBus):
            mech_buses = [mech_buses for _ in range(n_thrust)]
        if mech_splitters is None or isinstance(mech_splitters, MechSplitter):
            mech_splitters = [mech_splitters for _ in range(n_thrust)]

        # Create mechanical power group
        mech_group: om.Group = arch.add_subsystem("mech", om.Group())

        # Define inputs
        _, input_map = collect_inputs(
            mech_group,
            [
                (THROTTLE_INPUT, None, np.tile(1.0, nn)),
                (DURATION_INPUT, "s", 1.0),
                (ACTIVE_INPUT, None, np.tile(1.0, nn)),
                (FLTCOND_RHO_INPUT, "kg/m**3", np.tile(1.225, nn)),
                (FLTCOND_TAS_INPUT, "m/s", np.tile(100.0, nn)),
                (WEIGHT_INPUT, "kg", 1.0),
            ],
            name="mech_in_collect",
        )

        # Create and add components
        fuel_flow_outputs = []
        weight_outputs = []
        electric_load_outputs = []
        power_rating_outputs = []
        for i, thrust_group in enumerate(thrust_groups):
            engine, motor, inverter, mech_splitter, mech_bus, = (
                engines[i],
                motors[i],
                inverters[i],
                mech_splitters[i],
                mech_buses[i],
            )

            if engine is None and motor is None:
                raise RuntimeError("Either engine or motor should be present for thrust group %d!" % (i + 1,))

            # Create group for mechanical power generation components for this specific thrust group
            mech_thrust_group: om.Group = mech_group.add_subsystem("mech%d" % (i + 1,), om.Group())
            shaft_power_out_param = None
            shaft_speed_out_param = None
            rated_power_out_param = None
            throttle_param = None

            if engine is not None and motor is not None:  # used usually for parallel hybrid
                if mech_bus is None or mech_splitter is None:
                    raise RuntimeError(
                        "engines and motors are added as inputs, therefore, "
                        "mech_buses and mech_splitters must be provided as well"
                    )

                # define design params for eng
                _, eng_input_map = collect_inputs(
                    mech_thrust_group,
                    [
                        ("eng_rating", "kW/kg", engine.power_rating),
                        ("eng_output_rpm", "rpm", engine.output_rpm),
                    ],
                    name="eng_in_collect",
                )

                eng = mech_thrust_group.add_subsystem(
                    engine.name,
                    SimpleTurboshaft(
                        num_nodes=nn,
                        psfc=engine.psfc * 1.68965774e-7,
                        weight_inc=engine.specific_weight,
                        weight_base=engine.base_weight,
                    ),
                )
                conv_mech = mech_thrust_group.add_subsystem(
                    'conversion_mech',
                    om.ExecComp(['output_power = power_to_weight_ratio*weight'],
                                weight={'value': 1.0, 'units': 'kg'},
                                power_to_weight_ratio={'value': 1.0, 'units': 'kW/kg'},
                                output_power={'value': 1.0, 'units': 'kW'})
                )
                weight_param = ".".join([mech_thrust_group.name, conv_mech.name, "weight"])
                mech_group.connect(input_map[WEIGHT_INPUT], weight_param)
                mech_thrust_group.connect(eng_input_map["eng_rating"], conv_mech.name + ".power_to_weight_ratio")

                mech_thrust_group.connect(conv_mech.name + ".output_power", eng.name + ".shaft_power_rating")

                fuel_flow_outputs += [".".join([mech_thrust_group.name, eng.name, "fuel_flow"])]
                weight_outputs += [".".join([mech_thrust_group.name, eng.name, "component_weight"])]



                # define design params for motor
                _, mot_input_map = collect_inputs(
                    mech_thrust_group,
                    [
                        ("motor_rating", "kW/kg", motor.power_rating),
                        ("motor_output_rpm", "rpm", motor.output_rpm),
                        ("motor_efficiency", None, motor.efficiency),
                    ],
                    name="motor_in_collect",
                )

                # Add electric motor component
                mot = mech_thrust_group.add_subsystem(
                    motor.name,
                    SimpleMotor(
                        efficiency=motor.efficiency,
                        num_nodes=nn,
                        weight_inc=motor.specific_weight,
                        weight_base=motor.base_weight,
                        cost_inc=motor.cost_inc,
                        cost_base=motor.cost_base,
                    ),
                )
                conv_elec = mech_thrust_group.add_subsystem(
                    'conversion_elec',
                    om.ExecComp(['output_power = power_to_weight_ratio*weight'],
                                weight={'value': 1.0, 'units': 'kg'},
                                power_to_weight_ratio={'value': 1.0, 'units': 'kW/kg'},
                                output_power={'value': 1.0, 'units': 'kW'})
                )
                weight_param = ".".join([mech_thrust_group.name, conv_elec.name, "weight"])
                mech_group.connect(input_map[WEIGHT_INPUT], weight_param)
                mech_thrust_group.connect(mot_input_map["motor_rating"], conv_elec.name + ".power_to_weight_ratio")

                mech_thrust_group.connect(conv_elec.name + ".output_power", mot.name + ".elec_power_rating")

                weight_outputs += [".".join([mech_thrust_group.name, mot.name, "component_weight"])]
                if len(power_rating_outputs) == 0:
                    power_rating_outputs += [
                        ".".join([mech_thrust_group.name, conv_mech.name + ".output_power"]),
                        ".".join([mech_thrust_group.name, conv_elec.name + ".output_power"]),
                    ]

                if inverter is None:  # override if inverter is added
                    electric_load_outputs += [".".join([mech_thrust_group.name, mot.name, "elec_load"])]

                if i == 1:  # add OEI condition
                    # get propulsor active flag as scalar to use it for OEI
                    scalify_active_flag = ScalifyComponent(
                        vars=[
                            ("propulsor_active_vector", ACTIVE_INPUT + "_scalar", nn, None),
                        ]
                    )
                    mech_thrust_group.add_subsystem("scalify_active_input", subsys=scalify_active_flag)
                    mech_group.connect(
                        input_map[ACTIVE_INPUT],
                        mech_thrust_group.name + ".scalify_active_input" + ".propulsor_active_vector",
                    )

                    # add rated powers
                    sum_rated_power = om.ExecComp(
                        [
                            "tot_rated_power = engine_rated_power + active_flag * motor_rated_power * motor_eff",
                        ],
                        tot_rated_power={"val": 1, "units": "kW"},
                        engine_rated_power={"val": 1, "units": "kW"},
                        active_flag={"val": 1},
                        motor_eff={"val": 1},
                        motor_rated_power={"val": 1.0, "units": "kW"},
                    )
                    mech_thrust_group.add_subsystem("sum_rated_power", subsys=sum_rated_power)
                    mech_thrust_group.connect(conv_mech.name + ".output_power", "sum_rated_power" + ".engine_rated_power")
                    mech_thrust_group.connect(conv_elec.name + ".output_power", "sum_rated_power" + ".motor_rated_power")
                    mech_thrust_group.connect(mot_input_map["motor_efficiency"], "sum_rated_power" + ".motor_eff")
                    mech_thrust_group.connect(
                        "scalify_active_input" + "." + ACTIVE_INPUT + "_scalar", "sum_rated_power" + ".active_flag"
                    )
                else:  # all engines active condition
                    sum_rated_power = om.ExecComp(
                        [
                            "tot_rated_power = engine_rated_power + motor_rated_power * motor_eff",
                        ],
                        tot_rated_power={"val": 1, "units": "kW"},
                        engine_rated_power={"val": 1, "units": "kW"},
                        motor_eff={"val": 1},
                        motor_rated_power={"val": 1.0, "units": "kW"},
                    )
                    mech_thrust_group.add_subsystem("sum_rated_power", subsys=sum_rated_power)
                    mech_thrust_group.connect(conv_mech.name + ".output_power", "sum_rated_power" + ".engine_rated_power")
                    mech_thrust_group.connect(conv_elec.name + ".output_power", "sum_rated_power" + ".motor_rated_power")
                    mech_thrust_group.connect(mot_input_map["motor_efficiency"], "sum_rated_power" + ".motor_eff")

                # add rated powers of eng and motor for sizing
                sizing_rated_power = AddSubtractComp()
                sizing_rated_power.add_equation(
                    output_name="tot_rated_power",
                    input_names=[eng.name + "_rated_power", mot.name + "_rated_power"],
                    units="kW",
                )
                mech_thrust_group.add_subsystem("sizing_rated_power", subsys=sizing_rated_power)
                mech_thrust_group.connect(
                    conv_mech.name + ".output_power", "sizing_rated_power" + "." + eng.name + "_rated_power"
                )
                mech_thrust_group.connect(
                    conv_elec.name + ".output_power", "sizing_rated_power" + "." + mot.name + "_rated_power"
                )

                # get total shaft power output of (engine + motor) system
                get_shaft_power = om.ExecComp(
                    [
                        "tot_shaft_power = throttle_vec * total_rated_power",
                    ],
                    tot_shaft_power={"val": np.ones(nn), "units": "kW"},
                    throttle_vec={"val": np.ones(nn)},
                    total_rated_power={"val": 1.0, "units": "kW"},
                )
                mech_thrust_group.add_subsystem("eng_motor_shaft_power", subsys=get_shaft_power)
                mech_thrust_group.connect(
                    "sum_rated_power" + ".tot_rated_power", "eng_motor_shaft_power" + ".total_rated_power"
                )

                # define throttle parameter
                throttle_param = ".".join([mech_thrust_group.name, "eng_motor_shaft_power", "throttle_vec"])

                # add MechBus
                bus = mech_thrust_group.add_subsystem(
                    mech_bus.name, SimpleMechBus(num_nodes=nn, efficiency=mech_bus.efficiency, rpm_out=mech_bus.rpm_out)
                )
                mech_thrust_group.connect("eng_motor_shaft_power" + ".tot_shaft_power", bus.name + ".shaft_power_in")

                # add splitter
                # Define design params for splitter
                _, splitter_input_map = collect_inputs(
                    mech_thrust_group,
                    [
                        ("mech_DoH", None, mech_doh),
                    ],
                    name="splitter_in_collect",
                )

                # expand DoH to vector
                mech_thrust_group.add_subsystem(
                    "expand_DoH",
                    ExpandComponent(
                        vars=[
                            ("mech_DoH_scalar", "mech_DoH_vector", nn, None),
                        ]
                    ),
                )
                mech_thrust_group.connect(splitter_input_map["mech_DoH"], "expand_DoH" + ".mech_DoH_scalar")

                if i == 1:
                    # add get split fraction component for OEI
                    get_split_fraction = om.ExecComp(
                        [
                            "split_fraction_vec = active_flag_vec * mech_DoH_vec",
                        ],
                        split_fraction_vec={"val": np.zeros(nn)},
                        active_flag_vec={"val": np.zeros(nn)},
                        mech_DoH_vec={"val": np.ones(nn)},
                    )
                    mech_thrust_group.add_subsystem("get_split_fraction", subsys=get_split_fraction)
                    mech_thrust_group.connect("expand_DoH" + ".mech_DoH_vector", "get_split_fraction" + ".mech_DoH_vec")
                    mech_group.connect(
                        input_map[ACTIVE_INPUT], mech_thrust_group.name + ".get_split_fraction" + ".active_flag_vec"
                    )

                split = mech_thrust_group.add_subsystem(
                    mech_splitter.name,
                    PowerSplit(num_nodes=nn, efficiency=mech_splitter.efficiency, rule=mech_splitter.split_rule),
                )
                # add OEI condition
                if i == 1:  # use get split fraction to connect to split fraction
                    mech_thrust_group.connect(
                        "get_split_fraction" + ".split_fraction_vec", split.name + ".power_split_fraction"
                    )
                else:  # use input map to connect to split fraction
                    mech_thrust_group.connect("expand_DoH" + ".mech_DoH_vector", split.name + ".power_split_fraction")

                # connect shaft power input to splitter
                mech_thrust_group.connect("eng_motor_shaft_power" + ".tot_shaft_power", split.name + ".power_in")
                # track weight
                weight_outputs += [".".join([mech_thrust_group.name, split.name, "component_weight"])]

                # define required power outputs from motor and engine
                motor_req_power_out = ".".join([split.name, "power_out_A"])
                eng_req_power_out = ".".join([split.name, "power_out_B"])

                # define available power for engine and motor
                motor_avail_power_out = ".".join([mot.name, "shaft_power_out"])
                eng_avail_power_out = ".".join([eng.name, "shaft_power_out"])

                # find engine throttle
                throttle_from_power_balance(
                    group=mech_thrust_group,
                    power_req=eng_req_power_out,
                    power_avail=eng_avail_power_out,
                    units="kW",
                    comp_name=eng.name,
                    n=nn,
                )

                # find motor throttle
                throttle_from_power_balance(
                    group=mech_thrust_group,
                    power_req=motor_req_power_out,
                    power_avail=motor_avail_power_out,
                    units="kW",
                    comp_name=mot.name,
                    n=nn,
                )

                # define output parameters
                shaft_power_out_param = ".".join([mech_group.name, mech_thrust_group.name, bus.name, "shaft_power_out"])
                shaft_speed_out_param = ".".join([mech_group.name, mech_thrust_group.name, bus.name, "output_rpm"])
                rated_power_out_param = ".".join(
                    [mech_group.name, mech_thrust_group.name, "sizing_rated_power", "tot_rated_power"]
                )

            # Add turboshaft engine
            if engine is not None and motor is None:  # used usually for conventional architectures
                # Define design params
                _, eng_input_map = collect_inputs(
                    mech_thrust_group,
                    [
                        ("eng_rating", "kW/kg", engine.power_rating),
                        ("eng_output_rpm", "rpm", engine.output_rpm),
                    ],
                    name="eng_in_collect",
                )

                # add one engine inoperative case for prop systems with two or more engines
                if i == 1 and motor is None:  # if no of engines >=2, if yes, add a failed engine component to mech2
                    failedengine = ElementMultiplyDivideComp()
                    failedengine.add_equation(
                        "eng2throttle", input_names=["throttle_vec", "propulsor_active_flag"], vec_size=nn
                    )
                    failedengine = mech_thrust_group.add_subsystem("failedengine", failedengine)
                    mech_group.connect(
                        input_map[ACTIVE_INPUT], mech_thrust_group.name + ".failedengine" + ".propulsor_active_flag"
                    )

                # Add engine component
                eng = mech_thrust_group.add_subsystem(
                    engine.name,
                    SimpleTurboshaft(
                        num_nodes=nn,
                        psfc=engine.psfc * 1.68965774e-7,
                        weight_inc=engine.specific_weight,
                        weight_base=engine.base_weight,
                    ),
                )

                conv_mech = mech_thrust_group.add_subsystem(
                    'conversion_mech',
                    om.ExecComp(['output_power = power_to_weight_ratio*weight'],
                                weight={'value': 1.0, 'units': 'kg'},
                                power_to_weight_ratio={'value': 1.0, 'units': 'kW/kg'},
                                output_power={'value': 1.0, 'units': 'kW'})
                )
                weight_param = ".".join([mech_thrust_group.name, conv_mech.name, "weight"])
                mech_group.connect(input_map[WEIGHT_INPUT], weight_param)
                mech_thrust_group.connect(eng_input_map["eng_rating"], conv_mech.name + ".power_to_weight_ratio")


                mech_thrust_group.connect(conv_mech.name + ".output_power", eng.name + ".shaft_power_rating")

                fuel_flow_outputs += [".".join([mech_thrust_group.name, eng.name, "fuel_flow"])]
                weight_outputs += [".".join([mech_thrust_group.name, eng.name, "component_weight"])]
                if len(power_rating_outputs) == 0:
                    power_rating_outputs += [".".join([mech_thrust_group.name, conv_mech.name + ".output_power"])]

                # define out_params
                shaft_power_out_param = ".".join([mech_group.name, mech_thrust_group.name, eng.name, "shaft_power_out"])
                shaft_speed_out_param = ".".join(
                    [mech_group.name, mech_thrust_group.name, eng_input_map["eng_output_rpm"]]
                )
                rated_power_out_param = ".".join([mech_group.name, mech_thrust_group.name, conv_mech.name + ".output_power"])

                # define throttle parameter in case of one engine inoperative OEI or Normal
                if i == 1:  # in the case of OEI, for mech2, connect throttle to failedengine
                    throttle_param = ".".join([mech_thrust_group.name, "failedengine", "throttle_vec"])
                    mech_thrust_group.connect("failedengine" + ".eng2throttle", eng.name + ".throttle")
                else:  # Normal conditions
                    throttle_param = ".".join([mech_thrust_group.name, eng.name, "throttle"])

            # Add electric motor
            if motor is not None and engine is None:  # usually used for all electric, turboelectric, or series hybrid
                # Defined design params
                _, mot_input_map = collect_inputs(
                    mech_thrust_group,
                    [
                        ("motor_rating", "kW/kg", motor.power_rating),
                        ("motor_output_rpm", "rpm", motor.output_rpm),
                    ],
                    name="motor_in_collect",
                )

                # add one motor inoperative case for prop systems with two or more motors
                if i == 1:  # check if no of motors >=2, if yes, add a failed motor component to mech2 group
                    failedmotor = ElementMultiplyDivideComp()
                    failedmotor.add_equation(
                        "motor2throttle", input_names=["throttle_vec", "propulsor_active_flag"], vec_size=nn
                    )
                    failedmotor = mech_thrust_group.add_subsystem("failedmotor", failedmotor)
                    mech_group.connect(
                        input_map[ACTIVE_INPUT], mech_thrust_group.name + ".failedmotor" + ".propulsor_active_flag"
                    )

                # Add electric motor component
                mot = mech_thrust_group.add_subsystem(
                    motor.name,
                    SimpleMotor(
                        efficiency=motor.efficiency,
                        num_nodes=nn,
                        weight_inc=motor.specific_weight,
                        weight_base=motor.base_weight,
                        cost_inc=motor.cost_inc,
                        cost_base=motor.cost_base,
                    ),
                )
                conv_elec = mech_thrust_group.add_subsystem(
                    'conversion_elec',
                    om.ExecComp(['output_power = power_to_weight_ratio*weight'],
                                weight={'value': 1.0, 'units': 'kg'},
                                power_to_weight_ratio={'value': 1.0, 'units': 'kW/kg'},
                                output_power={'value': 1.0, 'units': 'kW'})
                )
                weight_param = ".".join([mech_thrust_group.name, conv_elec.name, "weight"])
                mech_group.connect(input_map[WEIGHT_INPUT], weight_param)
                mech_thrust_group.connect(mot_input_map["motor_rating"], conv_elec.name + ".power_to_weight_ratio")

                mech_thrust_group.connect(conv_elec.name + ".output_power", mot.name + ".elec_power_rating")

                weight_outputs += [".".join([mech_thrust_group.name, mot.name, "component_weight"])]
                if len(power_rating_outputs) == 0:
                    power_rating_outputs += [".".join([mech_thrust_group.name, conv_elec.name + ".output_power"])]

                if inverter is None:  # set electric load to motor load if no inverter is present
                    electric_load_outputs += [".".join([mech_thrust_group.name, mot.name, "elec_load"])]

                # define out_params
                shaft_power_out_param = ".".join([mech_group.name, mech_thrust_group.name, mot.name, "shaft_power_out"])
                shaft_speed_out_param = ".".join(
                    [mech_group.name, mech_thrust_group.name, mot_input_map["motor_output_rpm"]]
                )
                rated_power_out_param = ".".join(
                    [mech_group.name, mech_thrust_group.name, conv_elec.name + ".output_power"]
                )

                # define throttle parameter in case of one motor inoperative OEI or Normal
                if i == 1:  # in the case of OEI, for mech2, connect throttle to failedmotor
                    throttle_param = ".".join([mech_thrust_group.name, "failedmotor", "throttle_vec"])
                    mech_thrust_group.connect("failedmotor" + ".motor2throttle", mot.name + ".throttle")
                else:  # Normal conditions
                    throttle_param = ".".join([mech_thrust_group.name, mot.name, "throttle"])

            if inverter is not None:
                if motor is None:
                    raise RuntimeError("Inverter is added but no Motor!")
                else:
                    # inverter does not need in_collect, it has no inputs, only options
                    invert = mech_thrust_group.add_subsystem(
                        inverter.name,
                        SimpleConverterInverted(
                            num_nodes=nn,
                            efficiency=inverter.efficiency,
                            weight_inc=inverter.specific_weight,
                            weight_base=inverter.base_weight,
                            cost_inc=inverter.cost_inc,
                            cost_base=inverter.cost_base,
                        ),
                    )

                    weight_outputs += [".".join([mech_thrust_group.name, invert.name, "component_weight"])]
                    # override electric load to get it from inverter
                    electric_load_outputs += [".".join([mech_thrust_group.name, invert.name, "elec_power_in"])]

                    mech_thrust_group.connect(conv_elec.name + ".output_power", invert.name + ".elec_power_rating")
                    mech_thrust_group.connect(mot.name + ".elec_load", invert.name + ".elec_power_out")

            # Connect throttle input
            mech_group.connect(input_map[THROTTLE_INPUT], throttle_param)

            # Connect output shaft power to thrust generation group
            if shaft_power_out_param is None:
                raise RuntimeError("No shaft power generated for thrust group %d!" % (i + 1,))
            arch.connect(shaft_power_out_param, thrust_group.name + "." + SHAFT_POWER_INPUT)
            arch.connect(shaft_speed_out_param, thrust_group.name + "." + SHAFT_SPEED_INPUT)
            arch.connect(rated_power_out_param, thrust_group.name + "." + RATED_POWER_INPUT)

        # Calculate output sums

        create_output_sum(mech_group, FUEL_FLOW_OUTPUT, fuel_flow_outputs, "kg/s", n=nn)
        create_output_sum(mech_group, WEIGHT_OUTPUT, weight_outputs, "kg")
        create_output_sum(mech_group, ELECTRIC_POWER_OUTPUT, electric_load_outputs, "kW", n=nn)
        create_output_sum(mech_group, POWER_RATING_OUTPUT, power_rating_outputs, "kW")

        # Determine whether electric power generation is needed
        electric_power_needed = len(electric_load_outputs) > 0

        return mech_group, electric_power_needed
