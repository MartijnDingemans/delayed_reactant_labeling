Design Choices
==============
For both the predict and optimize modules it was essential to make the code as efficient as possible, while remaining
robust.

DataFrame Libraries
-------------------
The :func:`predict.DRL.predict_concentration` function returns polars.DataFrames instead of pandas.DataFrames as they
turned out to be alot more efficient to calculate the DRL curves and errors with. However, pandas series are more
convenient to print, and manipulate, as they behave more like dictionaries. Furthermore pandas objects give more
flexibility to store the data as JSON files. This is why two different dataframe libraries are used simultaneously.


Predicting Concentrations
-------------------------
Upon initialization of the :class:`predict.DRL` class, it analyzes the reaction which were inputted, and per reaction
it extracts the following data:

#. The value of the rate constant
#. The indices of each reactant
#. The indices of each product

The indices are related to their alphabetical order, and upon calling the :func:`predict.DRL.predict_concentration` it
rearranges the inputted concentration data to match with the internal reference. It does this by looking at the keys
of each concentration, so that misalignment cannot happen. However, as a consequence, the outputted data is sorted
in the same manner as the internal reference, and not according to the input of the initial concentration!

To calculate the predicted concentration at each time stamp the class implements the explicit Euler method, as it is a
straightforward algorithm to implement. To calculate the concentration for the next time stamp it loops over all reactions,
and per reaction:

#. It calculates the amount of created chemicals, according to :math:`\Delta t \cdot k \cdot \prod [reactants]`
#. Subtract the created amount from each reaction reactant.
#. Add the created amount to each reaction product.

Its important to note that changes are made to a copy of the concentrations array, so that the reactions all act on the
same data. We repeat this process the number of ``steps_per_step`` times, and discard the intermediate results. The last
array of predicted concentration is saved at the corresponding time stamp. Furthermore this function which calcules the
explicit Euler formula is just in time compiled using numba, allowing it to gain a large performance boost.

