Implementation details
======================
For both the predict and optimize modules it was essential to make the code as efficient as possible, while remaining
robust.

Rate equations
--------------
To be able to calculate the rate of change per chemical as a function of the current concentrations in the system,
the :class:`predict.DRL` class analyzes the reactions. The rate equation of each chemical can be decomposed into the
rate equations caused by each reaction step in the system. These reaction steps are what the user inputs into the model.

.. math::

    \frac{dc}{dt} = \sum_{r}{ \frac{dc_r}{dt} }

For each reaction we can calculate the amount of created chemical:

.. math::

    dc_r = k_r \cdot \prod{[q_r]}

where, :math:`k` is the rate constant for reaction :math:`r` with reactants :math:`q`. We can subtract this amount of created chemical from each reactant, and
add it to each product to get the rate of change.

.. _Jacobian:

Estimating the Jacobian
-----------------------
The Jacobian matrix is a matrix containing the partial derivatives of the rate equations of chemical :math:`i` with
respect to each chemical :math:`j`. Similarly to how we analyzed the rate equations we can again decompose the entire
model into individual reaction steps:

.. math::

    J_{i, j} =  \frac{\delta i}{\delta j} = \sum_{r}{\frac{\delta i_r}{\delta j}}

For each reaction we than calculate the derivative with respect to each reactant. This is because the rate equation
due to this reaction is by definition the rate constant multiplied by the concentration of each reactant, and the derivative
with respect to non-reactants is zero. To calculate the derivative we than take the product of the concentrations of all
reactants, :math:`q`, except the reactant whose derivative we take.

.. math::

    \frac{\delta i_r}{\delta j} = k_r \cdot \prod^{q_r}_{q_r \ne j}{[q_r]}

Subsequently we multiply this with the rate constant, :math:`k`, and add this to all reaction products, whereas
we subtract it from each reactant. Because we take a very simple approach to calculating the derivative, this method only
works for reaction steps which are first order in each chemical.

DataFrame Libraries
-------------------
The :func:`predict.DRL.predict_concentration` function returns polars.DataFrames instead of pandas.DataFrames as they
turned out to be alot more efficient to calculate the DRL curves and errors with. However, pandas series are more
convenient to print, and manipulate, as they behave more like dictionaries. Furthermore pandas objects give more
flexibility to store the data as JSON files. This is why two different dataframe libraries are used simultaneously.

.. _rate_equations:

Explict Euler formula
---------------------
The explicit Euler formula takes the rate of changes as calculated above, and adds it to the currently known concentrations.
It repeats this the number of ``steps_per_step`` times, and discards the intermediate results. The last
array of predicted concentration is saved at the corresponding time stamp.

This method does not work well in stiff problems and using an ODE solver is recommended.


