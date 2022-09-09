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

from __future__ import division
import numpy as np
from openmdao.api import ExplicitComponent
from openmdao.api import Group


class SimpleConverter(ExplicitComponent):
    """
    A simple converter changes the electric current passing between components from DC to AC and vica versa.
    It operates as either a rectifier or an inverter based on simple assumption of specific mass and constant
    efficiency.

    Assumptions
    ------
        1- constant efficiency applied on electric power input to provide power output
        2- constant specific mass in [kW/kg] used for component weight sizing

    Source
    ------
        Strack, M., Chiozzotto, G.P., Iwanizki, M., Plohr, M., Kuhn, M., “Conceptual Design Assessment of Advanced
        Hybrid Electric Turboprop Aircraft Configurations,” 17th AIAA Aviation Technology, Integration, and Operations
        Conference, AIAA AVIATION Forum, AIAA 2017-3068, Denver, CO, 2017
        Page 2. Table 2.

    Inputs
    ------
    elec_power_in : float
        Shaft power in to the converter (vector, W)
    elec_power_rating: float
        Electric (not mech) design power (scalar, W), usually equals the power of the component providing the electric
        power

    Outputs
    -------
    elec_power_out : float
        Electric power output from the converter (vector, W)
    heat_out : float
        Waste heat produced (vector, W)
    component_cost : float
        Nonrecurring cost of the component (scalar, USD)
    component_weight : float
        Weight of the component (scalar, kg)
    component_sizing_margin : float
        Equal to 1 when producing full rated power (vector, dimensionless)

    Options
    -------
    num_nodes : int
        Number of analysis points to run (sets vec length; default 1)
    efficiency : float
        Shaft power efficiency. Sensible range 0.0 < eta <= 1.0 (default 1)
    weight_inc : float
        Weight per unit rated power (default 1/5000, kg/W)
    weight_base : float
        Base weight (default 0, kg)
    cost_inc : float
        Cost per unit rated power (default 0.134228, USD/W)
    cost_base : float
        Base cost (default 1 USD) B

    """

    def initialize(self):
        # define technology factors
        self.options.declare('num_nodes', default=1, desc='Number of flight/control conditions')
        self.options.declare('efficiency', default=0.95, desc='Efficiency (dimensionless)')  # 95% efficiency
        self.options.declare('weight_inc', default=1/(10*1000), desc='kg/W')  # 10kW/kg
        self.options.declare('weight_base', default=0., desc='kg base weight')
        self.options.declare('cost_inc', default=100.0/745.0, desc='$ cost per watt')  # same as motor in OpenConcept
        self.options.declare('cost_base', default=1., desc='$ cost base')

    def setup(self):
        nn = self.options['num_nodes']
        self.add_input('elec_power_in', units='W', desc='Input electric power', shape=(nn,))
        self.add_input('elec_power_rating', units='W', desc='Rated output power')

        # outputs and partials
        eta_converter = self.options['efficiency']
        weight_inc = self.options['weight_inc']
        weight_base = self.options['weight_base']
        cost_inc = self.options['cost_inc']
        cost_base = self.options['cost_base']

        self.add_output('elec_power_out', units='W', desc='Output electric power', shape=(nn,))
        self.add_output('heat_out', units='W', desc='Waste heat out', shape=(nn,))
        self.add_output('component_cost', units='USD', desc='Generator component cost')
        self.add_output('component_weight', units='kg', desc='Generator component weight')
        self.add_output('component_sizing_margin', desc='Fraction of rated power', shape=(nn,))

        self.declare_partials('elec_power_out', 'elec_power_in',
                              val=eta_converter * np.ones(nn), rows=range(nn), cols=range(nn))
        self.declare_partials('heat_out', 'elec_power_in',
                              val=(1 - eta_converter) * np.ones(nn), rows=range(nn), cols=range(nn))
        self.declare_partials('component_cost', 'elec_power_rating', val=cost_inc)
        self.declare_partials('component_weight', 'elec_power_rating', val=weight_inc)
        self.declare_partials('component_sizing_margin', 'elec_power_in',
                              rows=range(nn), cols=range(nn))
        self.declare_partials('component_sizing_margin', 'elec_power_rating')

    def compute(self, inputs, outputs):

        eta_converter = self.options['efficiency']
        weight_inc = self.options['weight_inc']
        weight_base = self.options['weight_base']
        cost_inc = self.options['cost_inc']
        cost_base = self.options['cost_base']

        outputs['elec_power_out'] = inputs['elec_power_in'] * eta_converter
        outputs['heat_out'] = inputs['elec_power_in'] * (1 - eta_converter)
        outputs['component_cost'] = inputs['elec_power_rating'] * cost_inc + cost_base
        outputs['component_weight'] = inputs['elec_power_rating'] * weight_inc + weight_base
        outputs['component_sizing_margin'] = (inputs['elec_power_in'] *
                                              eta_converter / inputs['elec_power_rating'])

    def compute_partials(self, inputs, J):

        eta_converter = self.options['efficiency']
        J['component_sizing_margin', 'elec_power_in'] = eta_converter / inputs['elec_power_rating']
        J['component_sizing_margin', 'elec_power_rating'] = - (eta_converter * inputs['elec_power_in'] /
                                                               inputs['elec_power_rating'] ** 2)


class SimpleConverterInverted(ExplicitComponent):
    """
    A simple converter changes the electric current passing between components from DC to AC and vica versa.
    It operates as either a rectifier or an inverter based on simple assumption of specific mass and constant
    efficiency.

    "SimpleConverterInverted" differs from "SimpleConverter" in that it performs the computation backward, so based on
    the power out from the converter, an efficiency is applied to calculate the power input that should be passed to
    the converter. In summary, the "SimpleConverterInverted" component takes its power output as an input and
    calculates its power input as an output.

    Assumptions
    ------
        1- constant efficiency applied on electric power input to provide power output
        2- constant specific mass in [kW/kg] used for component weight sizing

    Source
    ------
        Strack, M., Chiozzotto, G.P., Iwanizki, M., Plohr, M., Kuhn, M., “Conceptual Design Assessment of Advanced
        Hybrid Electric Turboprop Aircraft Configurations,” 17th AIAA Aviation Technology, Integration, and Operations
        Conference, AIAA AVIATION Forum, AIAA 2017-3068, Denver, CO, 2017
        Page 2. Table 2.

    Inputs
    ------
    elec_power_out : float
        Electric power output from the converter (vector, W)
    elec_power_rating: float
        Electric (not mech) design power (scalar, W), usually equals the power of the component providing the electric
        power such as electric motor

    Outputs
    -------
    elec_power_in : float
        elec power in to the converter (vector, W)
    heat_out : float
        Waste heat produced (vector, W)
    component_cost : float
        Nonrecurring cost of the component (scalar, USD)
    component_weight : float
        Weight of the component (scalar, kg)
    component_sizing_margin : float
        Equal to 1 when producing full rated power (vector, dimensionless)

    Options
    -------
    num_nodes : int
        Number of analysis points to run (sets vec length; default 1)
    efficiency : float
        Shaft power efficiency. Sensible range 0.0 < eta <= 1.0 (default 1)
    weight_inc : float
        Weight per unit rated power (default 1/5000, kg/W)
    weight_base : float
        Base weight (default 0, kg)
    cost_inc : float
        Cost per unit rated power (default 0.134228, USD/W)
    cost_base : float
        Base cost (default 1 USD) B

    """

    def initialize(self):
        # define technology factors
        self.options.declare('num_nodes', default=1, desc='Number of flight/control conditions')
        self.options.declare('efficiency', default=0.95, desc='Efficiency (dimensionless)')  # 95% efficiency
        self.options.declare('weight_inc', default=1/(10*1000), desc='kg/W')  # 10kW/kg
        self.options.declare('weight_base', default=0., desc='kg base weight')
        self.options.declare('cost_inc', default=100.0/745.0, desc='$ cost per watt')  # same as motor in OpenConcept
        self.options.declare('cost_base', default=1., desc='$ cost base')

    def setup(self):
        nn = self.options['num_nodes']
        self.add_input('elec_power_out', units='W', desc='Input electric power', shape=(nn,))
        self.add_input('elec_power_rating', units='W', desc='Rated output power')

        # outputs and partials
        eta_converter = self.options['efficiency']
        weight_inc = self.options['weight_inc']
        weight_base = self.options['weight_base']
        cost_inc = self.options['cost_inc']
        cost_base = self.options['cost_base']

        self.add_output('elec_power_in', units='W', desc='Output electric power', shape=(nn,))
        self.add_output('heat_out', units='W', desc='Waste heat out', shape=(nn,))
        self.add_output('component_cost', units='USD', desc='Generator component cost')
        self.add_output('component_weight', units='kg', desc='Generator component weight')
        self.add_output('component_sizing_margin', desc='Fraction of rated power', shape=(nn,))

        self.declare_partials('elec_power_in', 'elec_power_out',
                              val=1/eta_converter * np.ones(nn), rows=range(nn), cols=range(nn))
        self.declare_partials('heat_out', 'elec_power_out',
                              val=((1/eta_converter) - 1) * np.ones(nn), rows=range(nn), cols=range(nn))
        self.declare_partials('component_cost', 'elec_power_rating', val=cost_inc)
        self.declare_partials('component_weight', 'elec_power_rating', val=weight_inc)
        self.declare_partials('component_sizing_margin', 'elec_power_out',
                              rows=range(nn), cols=range(nn))
        self.declare_partials('component_sizing_margin', 'elec_power_rating')

    def compute(self, inputs, outputs):

        eta_converter = self.options['efficiency']
        weight_inc = self.options['weight_inc']
        weight_base = self.options['weight_base']
        cost_inc = self.options['cost_inc']
        cost_base = self.options['cost_base']

        # check test: efficiency can't be zero or less
        if eta_converter <= 0:
            raise ValueError("Converter efficiency must be 0<eta<=1")
        outputs['elec_power_in'] = inputs['elec_power_out']/eta_converter
        outputs['heat_out'] = ((1/eta_converter) - 1) * inputs['elec_power_out']
        outputs['component_cost'] = inputs['elec_power_rating'] * cost_inc + cost_base
        outputs['component_weight'] = inputs['elec_power_rating'] * weight_inc + weight_base
        outputs['component_sizing_margin'] = (inputs['elec_power_out'] / inputs['elec_power_rating'])

    def compute_partials(self, inputs, J):

        eta_converter = self.options['efficiency']
        J['component_sizing_margin', 'elec_power_out'] = 1 / inputs['elec_power_rating']
        J['component_sizing_margin', 'elec_power_rating'] = -inputs['elec_power_out']/(inputs['elec_power_rating'] ** 2)
