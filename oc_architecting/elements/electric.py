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
from oc_architecting.elements.mech import *

from openconcept.components.battery import SOCBattery
from openconcept.components import SimpleTurboshaft, SimpleGenerator, PowerSplit
from oc_architecting.components import SimpleConverter, SimpleDCBusInverted

__all__ = [
    "ElectricPowerElements",
    "DCBus",
    "Batteries",
    "Engine",
    "Generator",
    "Rectifier",
    "DCEngineChain",
    "ACEngineChain",
    "ElecSplitter",
    "FUEL_FLOW_OUTPUT",
    "SOC_OUTPUT",
]

SOC_OUTPUT = "SOC"


@dataclass(frozen=False)
class DCBus(ArchElement):
    name: str = "dc_bus"

    efficiency: float = 0.99  #


@dataclass(frozen=False)
class ElecSplitter(ArchElement):
    """power splitter to divide a power input to two outputs A and B based on a split fraction and efficiency loss"""

    name: str = "elec_splitter"

    power_rating: float = 99999999  # 'W', maximum power rating of split component
    efficiency: float = 1.0  # always keep as 1, apply required efficiency in DC Bus component
    split_rule: str = "fraction"  # this sets the rule to always use a fraction between 0 and 1
    elec_DoH: float = 0.5  # degree of hybridization between Battery pack and DCEngineChains, 0 =< elec_DoH =< 1


@dataclass(frozen=False)
class Batteries(ArchElement):
    """Battery pack."""

    name: str = "battery"

    weight: float = 1000.0  # kg
    specific_power: float = 5000  # W/kg
    specific_energy: float = 300  # W/kg
    efficiency: float = 0.97
    cost_inc: float = 50.0  # $ per kg
    cost_base: float = 1.0  # $ per base


@dataclass(frozen=False)
class Generator(ArchElement):
    """An AC electric power generator."""

    name: str = "generator"

    efficiency: float = 0.97
    # power_rating: float = 260.  # kW, passed from engine inside the engine chain
    specific_weight: float = 1.0 / 5000  # kg/kW
    base_weight: float = 0.0  # kg
    cost_inc: float = 100.0 / 745.0  # $ per watt
    cost_base: float = 1.0  # $ per base


@dataclass(frozen=False)
class Rectifier(ArchElement):
    """An AC to DC rectifier."""

    name: str = "rectifier"

    efficiency: float = 0.97
    # power_rating: float = 260.  # kW, passed from engine inside the engine chain
    specific_weight: float = 1.0 / (10 * 1000)  # kg/kW
    base_weight: float = 0.0  # kg
    cost_inc: float = 100.0 / 745.0  # $ per watt
    cost_base: float = 1.0  # $ per base


DCEngineChain = Tuple[Engine, Generator, Rectifier]
ACEngineChain = Tuple[Engine, Generator]


@dataclass(frozen=False)
class ElectricPowerElements(ArchSubSystem):
    """Electrical power generation elements in the propulsion system architecture."""

    dc_bus: Optional[DCBus] = None
    splitter: Optional[ElecSplitter] = None
    batteries: Union[Batteries, List[Batteries]] = None
    engines_dc: Union[DCEngineChain, List[DCEngineChain]] = None
    engines_ac: Union[ACEngineChain, List[ACEngineChain]] = None
    elec_doh: Optional[Dict[str, float]] = None

    def get_dv_defs(self) -> List[Tuple[str, List[str], str, Any]]:

        elec_dvs = []
        if self.batteries is not None and type(self.batteries) != list:
            battery_paths = ["elec.battery_weight"]
            elec_dvs += (("ac|weights|W_battery", battery_paths, "kg", self.batteries.weight),)

        if self.engines_dc is not None and type(self.engines_dc) != list:
            elec_eng_paths = ["elec.eng_rating"]
            eng, gen, rect = self.engines_dc
            elec_dvs += (("ac|propulsion|elec_engine|rating", elec_eng_paths, "kW", eng.power_rating),)

        # if self.splitter is not None:
        #     elec_doh_paths = ['elec.elec_DoH']
        #     elec_dvs += ('ac|propulsion|elec_splitter|elec_DoH', elec_doh_paths, None, self.splitter.elec_DoH),

        return elec_dvs

    def create_electric_group(
        self, arch: om.Group, mech_power_group: om.Group, thrust_groups: List[om.Group], nn: int, phase: str
    ) -> om.Group:
        """
        Creates the electrical power generation group.

        Inputs: DURATION_INPUT, FLTCOND_RHO_INPUT[nn], FLTCOND_TAS_INPUT[nn]
        Outputs: FUEL_FLOW_OUTPUT[nn], WEIGHT_OUTPUT, SOC_OUTPUT[nn]
        """

        # Prepare components
        dc_bus = self.dc_bus
        splitter = self.splitter
        batteries = self.batteries
        engine_chains_dc = self.engines_dc
        engine_chains_ac = self.engines_ac
        elec_doh = (self.elec_doh or {}).get(phase, 1.)

        # check inputs
        # check batteries input
        if type(batteries) == list:
            if len(batteries) == 1:
                batteries = batteries[0]  # take first item of the list
            elif len(batteries) > 1:
                raise NotImplementedError("multiple battery pack design is not implemented yet")
            else:
                raise ValueError("Battery pack list cannot be empty")

        # check engine_chains_dc input
        if type(engine_chains_dc) == list:
            if len(engine_chains_dc) == 1:
                engine_chains_dc = engine_chains_dc[0]  # take first item of the list
            elif len(engine_chains_dc) > 1:
                raise NotImplementedError("multiple DC engine chains is not implemented yet")
            else:
                raise ValueError("engine_chains_dc list cannot be empty")

        # check engine_chains_ac input
        if type(engine_chains_ac) == list:
            if len(engine_chains_ac) == 1:
                engine_chains_ac = engine_chains_ac[0]  # take first item of the list
            elif len(engine_chains_ac) > 1:
                raise NotImplementedError("multiple AC engine chains is not implemented yet")
            else:
                raise ValueError("engine_chains_ac list cannot be empty")

        # Create electrical power group
        elec_group: om.Group = arch.add_subsystem("elec", om.Group())

        # Define inputs
        _, input_map = collect_inputs(
            elec_group,
            [
                (DURATION_INPUT, "s", 1.0),
                (FLTCOND_RHO_INPUT, "kg/m**3", np.tile(1.225, nn)),
                (FLTCOND_TAS_INPUT, "m/s", np.tile(100.0, nn)),
            ],
            name="elec_in_collect",
        )

        elec_load_input = None
        fuel_flow_outputs = []
        weight_outputs = []
        soc_outputs = []

        if dc_bus is not None:
            bus = elec_group.add_subsystem(dc_bus.name, SimpleDCBusInverted(num_nodes=nn, efficiency=dc_bus.efficiency))
            elec_load_input = bus.name + ".elec_power_out"  # connect elec load to dc bus

        if splitter is not None:
            if dc_bus is None:
                raise RuntimeError("dc_bus is required with a splitter")
            else:
                if engine_chains_dc is None or batteries is None:
                    raise RuntimeError("both engines and batteries are required with splitter in ElectricPowerElements")
                else:
                    # Define design params for splitter
                    _, splitter_input_map = collect_inputs(
                        elec_group,
                        [
                            ("elec_DoH", None, elec_doh),
                        ],
                        name="splitter_in_collect",
                    )

                    # expand DoH to vector
                    elec_group.add_subsystem(
                        "expand_DoH",
                        ExpandComponent(
                            vars=[
                                ("elec_DoH_scalar", "elec_DoH_vector", nn, None),
                            ]
                        ),
                    )
                    elec_group.connect(splitter_input_map["elec_DoH"], "expand_DoH" + ".elec_DoH_scalar")

                    split = elec_group.add_subsystem(
                        splitter.name,
                        PowerSplit(num_nodes=nn, efficiency=splitter.efficiency, rule=splitter.split_rule),
                    )

                    elec_group.connect("expand_DoH" + ".elec_DoH_vector", split.name + ".power_split_fraction")
                    elec_group.connect(bus.name + ".elec_power_in", split.name + ".power_in")
                    # define require power outputs
                    battery_req_power_out = ".".join([split.name, "power_out_A"])
                    eng_chain_req_power_out = ".".join([split.name, "power_out_B"])

                    # track weight
                    weight_outputs += [".".join([split.name, "component_weight"])]

        # Create and add batteries
        if batteries is not None:
            # Define design params
            _, bat_input_map = collect_inputs(
                elec_group,
                [
                    ("battery_weight", "kg", batteries.weight),
                ],
                name="bat_in_collect",
            )

            # Add battery component
            bat = elec_group.add_subsystem(
                batteries.name,
                SOCBattery(
                    num_nodes=nn,
                    efficiency=batteries.efficiency,
                    specific_power=batteries.specific_power,
                    specific_energy=batteries.specific_energy,
                    cost_inc=batteries.cost_inc,
                    cost_base=batteries.cost_base,
                ),
            )

            weight_outputs += [bat_input_map["battery_weight"]]
            soc_outputs += [bat.name + ".SOC"]

            if dc_bus is None:
                elec_load_input = bat.name + ".elec_load"  # directly connect elec load to battery
                raise RuntimeWarning("motors AC power load should not be connected directly to Battery DC power supply")
            else:
                if splitter is None:
                    elec_group.connect(bus.name + ".elec_power_in", bat.name + ".elec_load")
                else:
                    elec_group.connect(battery_req_power_out, bat.name + ".elec_load")

            elec_group.connect(bat_input_map["battery_weight"], bat.name + ".battery_weight")
            # elec_group.connect(input_map[DURATION_INPUT], bat.name + '.duration')

        # Add conventional engine chains
        if engine_chains_dc is not None:
            # parse components
            engine, generator, rectifier = engine_chains_dc
            # Define design params
            _, eng_input_map = collect_inputs(
                elec_group,
                [
                    ("eng_rating", "kW", engine.power_rating),
                    ("eng_output_rpm", "rpm", engine.output_rpm),
                ],
                name="eng_in_collect",
            )

            # Add engine component
            eng = elec_group.add_subsystem(
                engine.name,
                SimpleTurboshaft(
                    num_nodes=nn,
                    psfc=engine.psfc * 1.68965774e-7,
                    weight_inc=engine.specific_weight,
                    weight_base=engine.base_weight,
                ),
            )
            # track weight and fuel
            fuel_flow_outputs += [".".join([eng.name, "fuel_flow"])]
            weight_outputs += [".".join([eng.name, "component_weight"])]

            elec_group.connect(eng_input_map["eng_rating"], eng.name + ".shaft_power_rating")

            # add generator component
            gen = elec_group.add_subsystem(
                generator.name,
                SimpleGenerator(
                    num_nodes=nn,
                    efficiency=generator.efficiency,
                    weight_inc=generator.specific_weight,
                    weight_base=generator.base_weight,
                    cost_inc=generator.cost_inc,
                    cost_base=generator.cost_base,
                ),
            )
            # track weight
            weight_outputs += [".".join([gen.name, "component_weight"])]
            # connect variables
            elec_group.connect(eng_input_map["eng_rating"], gen.name + ".elec_power_rating")
            elec_group.connect(eng.name + ".shaft_power_out", gen.name + ".shaft_power_in")

            # add rectifier component
            rect = elec_group.add_subsystem(
                rectifier.name,
                SimpleConverter(
                    num_nodes=nn,
                    efficiency=rectifier.efficiency,
                    weight_inc=rectifier.specific_weight,
                    weight_base=rectifier.base_weight,
                    cost_inc=rectifier.cost_inc,
                    cost_base=rectifier.cost_base,
                ),
            )

            # track weight
            weight_outputs += [".".join([rect.name, "component_weight"])]
            # connect variables
            elec_group.connect(eng_input_map["eng_rating"], rect.name + ".elec_power_rating")
            elec_group.connect(gen.name + ".elec_power_out", rect.name + ".elec_power_in")

            # available output power of engine chain
            eng_chain_avail_power_out = ".".join([rect.name, "elec_power_out"])

            if splitter is not None:
                # find eng throttle to provide required power by using a balancer
                throttle_from_power_balance(
                    group=elec_group,
                    power_req=eng_chain_req_power_out,
                    power_avail=eng_chain_avail_power_out,
                    units="kW",
                    comp_name=eng.name,
                    n=nn,
                )
            else:
                if dc_bus is not None and batteries is None:
                    eng_chain_req_power_out = bus.name + ".elec_power_in"
                    throttle_from_power_balance(
                        group=elec_group,
                        power_req=eng_chain_req_power_out,
                        power_avail=eng_chain_avail_power_out,
                        units="kW",
                        comp_name=eng.name,
                        n=nn,
                    )

        if engine_chains_ac is not None:
            raise NotImplementedError("AC architectures not implemented yet!")

        # Connect electric load
        if elec_load_input is None:
            raise RuntimeError("Cannot connect electric load!")
        elec_load_output_param = mech_power_group.name + "." + ELECTRIC_POWER_OUTPUT
        arch.connect(elec_load_output_param, elec_group.name + "." + elec_load_input)

        # Calculate output sums
        create_output_sum(elec_group, FUEL_FLOW_OUTPUT, fuel_flow_outputs, "kg/s", n=nn)
        create_output_sum(elec_group, WEIGHT_OUTPUT, weight_outputs, "kg")
        create_output_sum(elec_group, SOC_OUTPUT, soc_outputs, n=nn)

        return elec_group
