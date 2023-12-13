Delayed-reactant-labeling
=========================
**Delayed Reactant Labeling** (DRL) is a method which allows kinetic information to be extracted from experiments on a
mass spectrometer. By coupling the mass spectrometer to a TIMS unit, also different diastereomers can be modeled.

This module can:

i) extract the rate equations of a chemical system from the reaction steps,
ii) model the rate equations using a Radau ODE solver, to ensure accuracy and efficiency of the model,
iii) use the Nelder-Mead algorithm with adapted parameters to find an optimal solution in a highly dimensional space,
iv) perform multiple analyses of a system from different starting positions and,
v) visualize the model in different ways.

To get started check out the :doc:`installation` section for further information.

Contents
--------
.. toctree::
   :maxdepth: 3

   installation
   predict
   optimize
   visualize
   extensive_example
   design_choices
