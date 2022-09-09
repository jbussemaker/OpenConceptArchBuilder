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


class SimpleDCBus(ExplicitComponent):
    """
    A simple DC electric bus which represents an electric conductors that transfer electric power from source to
    destination

    Assumptions
    ------
        1- the weight of the bus is not considered
        2- constant efficiency of 99% by default

    Source
    ------
        Benjamin J. Brelje∗, Joaquim R.R.A. Martins, Electric, hybrid, and turboelectric fixed-wing aircraft: A review
        of concepts, models, and design approaches,2019.
        Page 11. Electrical system Architecture

        Jeffryes W. Chapman, Multi-point Design and Optimization of a Turboshaft Engine for Tiltwing Turboelectric VTOL
        Air Taxi, 2019.
        Page 5. Table 2.

    Inputs
    ------
    elec_power_in : float
        Shaft power in to the bus (vector, W)

    Outputs
    -------
    elec_power_out : float
        Electric power output from the bus (vector, W)
    heat_out : float
        Waste heat produced (vector, W)

    Options
    -------
    num_nodes : int
        Number of analysis points to run (sets vec length; default 1)
    efficiency : float
        component efficiency. Sensible range where 0.0 < efficiency <= 1.0 (default 1)

    """

    def initialize(self):
        # define technology factors
        self.options.declare('num_nodes', default=1, desc='Number of flight/control conditions')
        self.options.declare('efficiency', default=0.99, desc='Efficiency (dimensionless)')  # 99% efficiency

    def setup(self):
        nn = self.options['num_nodes']
        self.add_input('elec_power_in', units='W', desc='Input electric power', shape=(nn,))

        # outputs and partials
        eta_bus = self.options['efficiency']

        self.add_output('elec_power_out', units='W', desc='Output electric power', shape=(nn,))
        self.add_output('heat_out', units='W', desc='Waste heat out', shape=(nn,))

        self.declare_partials('elec_power_out', 'elec_power_in',
                              val=eta_bus * np.ones(nn), rows=range(nn), cols=range(nn))
        self.declare_partials('heat_out', 'elec_power_in',
                              val=(1 - eta_bus) * np.ones(nn), rows=range(nn), cols=range(nn))

    def compute(self, inputs, outputs):

        eta_bus = self.options['efficiency']

        outputs['elec_power_out'] = inputs['elec_power_in'] * eta_bus
        outputs['heat_out'] = inputs['elec_power_in'] * (1 - eta_bus)

    def compute_partials(self, inputs, J):

        pass


class SimpleDCBusInverted(ExplicitComponent):
    """
    A simple DC electric bus which represents an electric conductors that transfer electric power from source to
    destination

    "SimpleDCBusInverted" differs from "SimpleDCBus" in that it performs the computation backward, so based on
    the power out from the DC Bus, an efficiency is applied to calculate the power input that should passed to
    the DC Bus. In summary, the "SimpleDCBusInverted" component takes its power output as an input and
    calculates its power input as an output.

    Assumptions
    ------
        1- the weight of the bus is not considered
        2- constant efficiency of 99% by default

    Source
    ------
        Benjamin J. Brelje∗, Joaquim R.R.A. Martins, Electric, hybrid, and turboelectric fixed-wing aircraft: A review
        of concepts, models, and design approaches,2019.
        Page 11. Electrical system Architecture

        Jeffryes W. Chapman, Multi-point Design and Optimization of a Turboshaft Engine for Tiltwing Turboelectric VTOL
        Air Taxi, 2019.
        Page 5. Table 2.

    Inputs
    ------
    elec_power_out : float
        the elec power output of the bus (vector, W)

    Outputs
    -------
    elec_power_in : float
        Electric power that should pass into bus (vector, W)
    heat_out : float
        Waste heat produced (vector, W)

    Options
    -------
    num_nodes : int
        Number of analysis points to run (sets vec length; default 1)
    efficiency : float
        component efficiency. Sensible range where 0.0 < efficiency <= 1.0 (default 1)

    """

    def initialize(self):
        # define technology factors
        self.options.declare('num_nodes', default=1, desc='Number of flight/control conditions')
        self.options.declare('efficiency', default=0.99, desc='Efficiency (dimensionless)')  # 99% efficiency

    def setup(self):
        nn = self.options['num_nodes']
        self.add_input('elec_power_out', units='W', desc='Input electric power', shape=(nn,))

        # outputs and partials
        eta_bus = self.options['efficiency']

        self.add_output('elec_power_in', units='W', desc='Output electric power', shape=(nn,))
        self.add_output('heat_out', units='W', desc='Waste heat out', shape=(nn,))

        self.declare_partials('elec_power_in', 'elec_power_out',
                              val=1/eta_bus * np.ones(nn), rows=range(nn), cols=range(nn))
        self.declare_partials('heat_out', 'elec_power_out',
                              val=((1/eta_bus) - 1) * np.ones(nn), rows=range(nn), cols=range(nn))

    def compute(self, inputs, outputs):

        eta_bus = self.options['efficiency']

        if eta_bus <= 0:
            raise ValueError("DC Bus efficiency must be 0<eta<=1")

        outputs['elec_power_in'] = inputs['elec_power_out'] / eta_bus
        outputs['heat_out'] = ((1/eta_bus) - 1)*inputs['elec_power_out']

    def compute_partials(self, inputs, J):

        pass


class SimpleMechBus(ExplicitComponent):
    """
    A simple Mechanical bus which represents an efficiency loss due to combining the power of a motor and an engine.
    It takes a shaft power input (combined shaft power of motor and engine), applies an efficiency loss and gives
    a shaft power output at a predefined RPM. This component is only used in a parallel hybrid propulsion system

    Assumptions
    ------
        1- the weight of the bus is not considered
        2- constant efficiency of 95% by default

    Source
    ------
        Guillem Moreno Bravo, Nurgeldy Praliyev, Arpad Veress, Performance analysis of hybrid electric and distributed
        propulsion system applied on a light aircraft, 2021.
        Page 7. section 4.3 parallel configuration

    Inputs
    ------
    shaft_power_in : float
        Shaft power into the bus (vector, kW)

    Outputs
    -------
    shaft_power_out : float
        shaft power output from the bus (vector, kW)
    heat_out : float
        Waste heat produced (vector, kW)
    output_rpm: float
        a predefined output rpm of the mechanical bus (Scalar, rpm)

    Options
    -------
    num_nodes : int
        Number of analysis points to run (sets vec length; default 1)
    efficiency : float
        component efficiency. Sensible range where 0.0 < efficiency <= 1.0 (default 0.95)
    rpm_out : float
        to define the rpm output of the mechanical bus component (default 5500)

    """

    def initialize(self):
        # define technology factors
        self.options.declare('num_nodes', default=1, desc='Number of flight/control conditions')
        self.options.declare('efficiency', default=0.95, desc='Efficiency (dimensionless)')  # 95% efficiency
        self.options.declare('rpm_out', default=5500, desc='output rpm of the mechanical bus (rpm)')

    def setup(self):
        nn = self.options['num_nodes']
        self.add_input('shaft_power_in', units='kW', desc='Input shaft power', shape=(nn,))

        # outputs and partials
        eta_bus = self.options['efficiency']

        self.add_output('shaft_power_out', units='kW', desc='Output shaft power', shape=(nn,))
        self.add_output('heat_out', units='kW', desc='Waste heat out', shape=(nn,))
        self.add_output('output_rpm', units='rpm', desc='shaft speed output')

        self.declare_partials('shaft_power_out', 'shaft_power_in',
                              val=eta_bus * np.ones(nn), rows=range(nn), cols=range(nn))
        self.declare_partials('heat_out', 'shaft_power_in',
                              val=(1 - eta_bus) * np.ones(nn), rows=range(nn), cols=range(nn))

    def compute(self, inputs, outputs):

        eta_bus = self.options['efficiency']
        rpm_out = self.options['rpm_out']

        outputs['shaft_power_out'] = inputs['shaft_power_in'] * eta_bus
        outputs['heat_out'] = inputs['shaft_power_in'] * (1 - eta_bus)
        outputs['output_rpm'] = rpm_out

    def compute_partials(self, inputs, J):

        pass
