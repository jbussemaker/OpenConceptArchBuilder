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


class SimpleGearbox(Group):
    """
    A simple gearbox component is used for speed reduction.

    Assumptions
    ------
        1- efficiency correlation based on the ratio between input power to rating power, correlation developed based on
            an Allison T56/501 turboprop gearbox
        2- a statistical weight correlation for component sizing
        3- the model used gives best estimates for three to one power reduction ratio
        4- shaft speed input and output design values are required. shaft speed input can be in the order of few
           thousands, ex: 5500 rpm. shaft speed output must be in the order of few hundreds or more, ex: 1700 rpm.
           these two values are used only as sizing parameters to calculate the weight of the gearbox

        Note: Turboprops engines usually rotate at some 20,000-40,000 RPM, therefore, the gearbox is usually a standard
              part of the engine unit to provide low rpm output in the order of 2000s rpm. Check before adding a gearbox
              component for conventional turboprop propulsion system as it might be included in the engine weight
              estimation calculations.
              Source: GUDMUNDSSON, General Aviation Aircraft Design, Page 228
                      Certification Data sheet, PT6 engines, 2012.

    Source
    ------
        S. Stückl, Methods for the Design and Evaluation of Future Aircraft Concepts Utilizing Electric Propulsion, 2016
        Page 47, equations 38 and 39

        NASA Glenn Research Center Program in HighPower Density Motors for Aeropropulsion, 2005
        page 8, figure 9

    Inputs
    ------
    shaft_power_in : float
        Shaft power in to the gearbox (vector, kW)
    shaft_power_rating: float
        Mechanical shaft design power (scalar, kW), usually equals the rating power of the component providing the
        power, either an electric motor, a turboshaft or a mechanical bus, or receiving the power like a propeller
    shaft_speed_in: float
        the typical rotational speed at the input shaft for the gearbox (Scalar, RPM), comes from either the motor
        output rpm or the turboshaft output rpm or mech bus (motor+turboshaft) output rpm. It is only used as
        a sizing parameter in the weight correlation, i.e. weight of gearbox
        the shaft_speed_in should be selected as the typical operational rotational speed of the component
        ex ac|propulsion|motor|output_rpm = 6000 [rpm]
    shaft_speed_out: float
        the typical shaft speed output of the gearbox that will be delivered to the propulsive device (Scalar, RPM)
        at operation conditions. It is only used as a sizing parameter in the weight correlation, i.e. weight of gearbox
        Note: Please note that the model used is a statistical correlation which works well for values typical for
        fixed wing aircrafts or helicopters
        For the KingAir C90Gt propellers Operation, a value of 1750 rpm is recommended for best cruise performance
        source: flight manual, https://www.aso.com/seller/23680/176815/handbook.pdf
        ex ac|propulsion|propeller|design_rpm = 1750 [rpm]

    Outputs
    -------
    shaft_power_out : float
        shaft power output from the converter (vector, kW)
    heat_out : float
        Waste heat produced (vector, kW)
    component_weight : float
        Weight of the component (scalar, kg)

    Options
    -------
    num_nodes : int
        Number of analysis points to run (sets vec length; default 1)

    """

    def initialize(self):
        self.options.declare('num_nodes', default=1, desc='Number of flight/control conditions')

    def setup(self):
        nn = self.options['num_nodes']
        self.add_subsystem('GearboxEff', GearboxEfficiencyCalc(num_nodes=nn),
                           promotes_inputs=["shaft_power_in", 'shaft_power_rating'],
                           promotes_outputs=["efficiency"])
        self.add_subsystem('GearboxCalc', GearboxCalc(num_nodes=nn),
                           promotes_inputs=["efficiency", 'shaft_power_in', 'shaft_power_rating',
                                            'shaft_speed_in', 'shaft_speed_out'],
                           promotes_outputs=["shaft_power_out", 'heat_out', 'component_weight'])


class GearboxEfficiencyCalc(ExplicitComponent):
    """
    a helper class to calculate gearbox efficiency based on input power and rated power of gearbox
    """

    def initialize(self):
        self.options.declare('num_nodes', default=1, desc='Number of flight/control conditions')

    def setup(self):
        nn = self.options['num_nodes']
        self.add_input('shaft_power_in', units='kW', desc='shaft power input to the gearbox', shape=(nn,))
        self.add_input('shaft_power_rating', units='kW', desc='shaft power rating of the gearbox')
        self.add_output('efficiency', desc='gearbox efficiency', shape=(nn,))
        self.declare_partials('efficiency', 'shaft_power_in', rows=range(nn), cols=range(nn))
        self.declare_partials('efficiency', 'shaft_power_rating')

    def compute(self, inputs, outputs):
        outputs['efficiency'] = 0.989 * ((inputs['shaft_power_in'] / inputs['shaft_power_rating']) ** 0.0135)

    def compute_partials(self, inputs, J):
        nn = self.options['num_nodes']
        # check the value of shaft_power_in to avoid raising zero to negative number
        # without this check partial derivatives will result in a division by zero error when shaft_power_in = 0
        if np.equal(inputs['shaft_power_in'], np.zeros(nn)).all():
            J['efficiency', 'shaft_power_in'] = np.zeros(nn)
            J['efficiency', 'shaft_power_rating'] = np.zeros(nn)
        else:
            J['efficiency', 'shaft_power_in'] = 0.0135 * 0.989 * (
                    inputs['shaft_power_in'] / inputs['shaft_power_rating']) ** (0.0135 - 1) * (
                                                        1 / inputs['shaft_power_rating'])
            J['efficiency', 'shaft_power_rating'] = 0.0135 * 0.989 * (
                    inputs['shaft_power_in'] / inputs['shaft_power_rating']) ** (0.0135 - 1) * (
                                                            -inputs['shaft_power_in'] / (
                                                             inputs['shaft_power_rating'] ** 2))


class GearboxCalc(ExplicitComponent):
    """
    a helper class to calculate gearbox
        - shaft power output
        - heat output
        - weight

    """

    def initialize(self):
        self.options.declare('num_nodes', default=1, desc='Number of flight/control conditions')

    def setup(self):
        nn = self.options['num_nodes']

        self.add_input('efficiency', desc='gearbox efficiency', shape=(nn,))
        self.add_input('shaft_power_in', units='kW', desc='shaft power input to the gearbox', shape=(nn,))
        self.add_input('shaft_power_rating', units='kW', desc='shaft power rating of the gearbox')
        self.add_input('shaft_speed_in', units='rpm', val=5500,
                       desc='the rotational speed at the input shaft for the gearbox')
        self.add_input('shaft_speed_out', units='rpm', val=1500,
                       desc='the rotational speed at the output shaft for the gearbox')

        self.add_output('shaft_power_out', units='kW', desc='Output shaft power', shape=(nn,))
        self.add_output('heat_out', units='kW', desc='heat loss', shape=(nn,))
        self.add_output('component_weight', units='kg', desc='weight of the gearbox')

        self.declare_partials('shaft_power_out', 'shaft_power_in',
                              rows=range(nn), cols=range(nn))
        self.declare_partials('shaft_power_out', 'efficiency',
                              rows=range(nn), cols=range(nn))

        self.declare_partials('heat_out', 'shaft_power_in',
                              rows=range(nn), cols=range(nn))
        self.declare_partials('heat_out', 'efficiency',
                              rows=range(nn), cols=range(nn))

        self.declare_partials('component_weight', 'shaft_power_rating')
        self.declare_partials('component_weight', 'shaft_speed_in')
        self.declare_partials('component_weight', 'shaft_speed_out')

    def compute(self, inputs, outputs):
        outputs['shaft_power_out'] = inputs['shaft_power_in'] * inputs['efficiency']
        outputs['heat_out'] = inputs['shaft_power_in'] * (1 - inputs['efficiency'])
        # use a multiplication factor of 26 for future applications, 34 for 2000s, and 43 for the year 1980s
        # Source: S. Stückl, Methods for the Design and Evaluation of Future Aircraft Concepts Utilizing Electric
        #         Propulsion, 2016. Page 47, equations 38
        outputs['component_weight'] = 26 * (inputs['shaft_power_rating'] ** 0.76) * (
                inputs['shaft_speed_in'] ** 0.13) / (inputs['shaft_speed_out'] ** 0.89)

    def compute_partials(self, inputs, J):
        J['shaft_power_out', 'shaft_power_in'] = inputs['efficiency']
        J['shaft_power_out', 'efficiency'] = inputs['shaft_power_in']

        J['heat_out', 'shaft_power_in'] = (1 - inputs['efficiency'])
        J['heat_out', 'efficiency'] = -inputs['shaft_power_in']

        J['component_weight', 'shaft_power_rating'] = 0.76 * 26 * (
                (inputs['shaft_speed_in'] ** 0.13) / (inputs['shaft_speed_out'] ** 0.89)) * (
                                                              inputs['shaft_power_rating'] ** (0.76 - 1)) * 1
        J['component_weight', 'shaft_speed_in'] = 0.13 * 26 * (
                (inputs['shaft_power_rating'] ** 0.76) / (inputs['shaft_speed_out'] ** 0.89)) * (
                                                          inputs['shaft_speed_in'] ** (0.13 - 1)) * 1
        J['component_weight', 'shaft_speed_out'] = -0.89 * 26 * ((inputs['shaft_power_rating'] ** 0.76) * (
                inputs['shaft_speed_in'] ** 0.13) * (inputs['shaft_speed_out'] ** (0.89 - 1))) / (
                                                           (inputs['shaft_speed_out'] ** 0.89) ** 2)
