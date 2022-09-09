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
from openmdao.core.component import Component
from openconcept.utilities.dvlabel import DVLabel

__all__ = [
    "create_output_sum",
    "collect_inputs",
    "collect_outputs",
    "ExpandComponent",
    "ScalifyComponent",
    "throttle_from_power_balance",
]


def create_output_sum(group: om.Group, output: str, inputs: List[str], units: str = None, n=1) -> Component:
    sum_params = ["sum_input_%d" % i for i in range(len(inputs))]

    sub_name = output + "_sum"
    if len(inputs) == 0:
        comp = group.add_subsystem(
            sub_name, om.IndepVarComp(output, val=np.tile(0.0, n), units=units), promotes_outputs=[output]
        )
    elif len(inputs) == 1:
        kwargs = {
            output: {"val": np.tile(0.0, n), "units": units},
            sum_params[0]: {"val": np.tile(0.0, n), "units": units},
        }
        comp = group.add_subsystem(
            sub_name,
            om.ExecComp(
                [
                    "%s = %s" % (output, sum_params[0]),
                ],
                **kwargs
            ),
            promotes_outputs=[output],
        )
    else:
        comp = group.add_subsystem(
            sub_name,
            om.AddSubtractComp(output_name=output, input_names=sum_params, units=units, vec_size=n),
            promotes_outputs=[output],
        )

    for i, param in enumerate(sum_params):
        group.connect(inputs[i], sub_name + "." + param)

    return comp


def collect_inputs(
    group: om.Group, inputs: List[Tuple[str, Optional[str], Optional[Union[float, List[float]]]]], name: str = None
) -> Tuple[DVLabel, Dict[str, str]]:
    label_map = {inp[0]: inp[0] + "_pass" for inp in inputs}

    comp = group.add_subsystem(
        name or "in_collect",
        DVLabel([[inp, label_map[inp], val, units] for inp, units, val in inputs]),
        promotes_inputs=["*"],
        promotes_outputs=["*"],
    )

    return comp, label_map


def collect_outputs(
    group: om.Group, outputs: List[Tuple[str, Optional[str]]], name: str = None
) -> Tuple[DVLabel, Dict[str, str]]:
    label_map = {out: out + "_pass" for out, _ in outputs}

    comp = group.add_subsystem(
        name or "out_collect",
        DVLabel([[label_map[out], out, 1.0, units] for out, units in outputs]),
        promotes_inputs=["*"],
        promotes_outputs=["*"],
    )

    return comp, label_map


def throttle_from_power_balance(
    group: om.Group,
    power_req: str = None,
    power_avail: str = None,
    units: str = None,
    comp_name: str = None,
    n: int = 1,
) -> Component:
    """
    a helper function that receives as input an OpenMDAO Group, that has a component for which it is required to
    find the throttle input to achieve power balance between two variables, power_req and power_avail. it is usually
    used to find the throttle value of a turboshaft engine or an electric motor that is needed to provide the
    required power.

    Inputs:
        group: om.Group which contains the component for which we want fo find the throttle
        power_req: the string the corresponds to the required power variable inside the group
        power_avail: the string that corresponds to the available power variable inside the group
        units: the units to be used to equate both power variables
        comp_name: the name of the component for which we want to find throttle input, the component must have
                    a thottle input variable with the name comp_name.throttle
        n: number of nodes such that the throttle will be in the form np.ones(n,)*value

    returns:
        an OpenMDAO BalanceComp with the name balancer_name

    changes:
        It adds a BalanceComp to the group and connects the balancer_throttle to the comp_name.throttle and sets the
        numerical model to solve for the throttle such that the power_req and power_avail are equal
    """
    balancer_name = comp_name + "_throttle_set"
    balancer_output = comp_name + "_throttle"
    balancer_rhs_name = "power_req"
    balancer_lhs_name = "power_avail"
    balancer = om.BalanceComp(
        balancer_output,
        val=np.ones((n,)) * 0.5,
        units=None,
        eq_units=units,
        rhs_name=balancer_rhs_name,
        lhs_name=balancer_lhs_name,
        lower=0.0,
        upper=1.5,
    )

    balancer_comp = group.add_subsystem(balancer_name, balancer)
    group.connect(power_req, balancer_name + "." + balancer_rhs_name)
    group.connect(power_avail, balancer_name + "." + balancer_lhs_name)
    balancer_throttle = balancer_name + "." + balancer_output
    comp_throttle = comp_name + ".throttle"
    group.connect(balancer_throttle, comp_throttle)

    return balancer_comp


class ExpandComponent(om.ExplicitComponent):
    """
    Component that expands a 1-dimensional input to n-dimensional output.
    """

    def initialize(self):
        self.options.declare(
            "vars", types=(list,), desc="Variables to expand: (input_name, output_name, n_output, units)"
        )

    def setup(self):
        for input_name, output_name, n_output, units in self.options["vars"]:
            self.add_input(input_name, val=1.0, units=units)
            self.add_output(output_name, val=np.ones((n_output,)), units=units)

        self.declare_partials("*", "*")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        for input_name, output_name, n_output, _ in self.options["vars"]:
            outputs[output_name] = np.ones((n_output,)) * inputs[input_name]

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        for input_name, output_name, n_output, _ in self.options["vars"]:
            partials[output_name, input_name] = np.ones((n_output,))


class ScalifyComponent(om.ExplicitComponent):
    """
    Component that scalifies an n-dimensional input to 1-dimensional output.
    input is a numpy array of length n where all members are the same
        Ex: input = np.ones(n)
            output = input[0]
    """

    def initialize(self):
        self.options.declare(
            "vars", types=(list,), desc="Variables to scalify: (input_name, output_name, n_input, units)"
        )

    def setup(self):
        for input_name, output_name, n_input, units in self.options["vars"]:
            self.add_input(input_name, val=np.ones((n_input,)), units=units)
            self.add_output(output_name, val=1.0, units=units)

        self.declare_partials("*", "*")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        for input_name, output_name, n_input, _ in self.options["vars"]:
            outputs[output_name] = inputs[input_name][0]

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        pass


if __name__ == "__main__":  # test ExpandComponent and ScalifyComponent

    nn = 11  # vector size
    model = om.Group()
    model.add_subsystem(
        "multiply",
        om.ExecComp("f_xy = x * y", f_xy={"units": "m*m"}, x={"units": "m"}, y={"units": "m"}),
        promotes_outputs=["*"],
    )

    model.add_subsystem(
        "expand",
        ExpandComponent(
            vars=[
                ("f_xy", "f_xy_vector", nn, "m*m"),
            ]
        ),
        promotes_inputs=["*"],
        promotes_outputs=["*"],
    )

    model.add_subsystem(
        "scalify",
        ScalifyComponent(
            vars=[
                ("f_xy_vector", "f_xy_scalar", nn, "m*m"),
            ]
        ),
        promotes_inputs=["*"],
    )

    prob = om.Problem(model)
    prob.setup()

    prob.set_val("multiply.x", 3.0)
    prob.set_val("multiply.y", 5.0)

    prob.run_model()
    print(prob.get_val("multiply.x"))
    print(prob.get_val("multiply.y"))
    print(prob.get_val("multiply.f_xy"))
    print(prob.get_val("expand.f_xy_vector"))
    print(prob.get_val("scalify.f_xy_scalar"))

    # prob.model.list_inputs(units=True,
    #                        prom_name=True,
    #                        shape=True,
    #                        hierarchical=False,
    #                        print_arrays=True)
    #
    # # show outputs
    # prob.model.list_outputs(implicit=True,
    #                         explicit=True,
    #                         prom_name=True,
    #                         units=True,
    #                         shape=True,
    #                         bounds=False,
    #                         residuals=False,
    #                         scaling=False,
    #                         hierarchical=False,
    #                         print_arrays=True)
