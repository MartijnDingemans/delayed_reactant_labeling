Implementation details
======================
For both the predict and optimize modules it was essential to make the code as efficient as possible, while remaining
robust.

.. _rate_equations:

Rate equations
--------------
To be able to calculate the rate of change per chemical as a function of the current concentrations in the system,
the :class:`predict.DRL` class analyzes the reactions. The rate equation of each chemical for the complete model
can be decomposed into the rate equations caused by each reaction step in the system.
These reaction steps are what the user inputs into the model.

.. math::

    \frac{dc}{dt} = \sum_{r}{ \frac{dc_r}{dt} }

We can first initialize an array for :math:`dc/dt` filled with zeros, subsequently loop over each reaction step, and
calculate the amount of created chemical(s) as follows:

.. math::

    \frac{dc_r}{dt} = k_r \prod{[q_r]}

where, :math:`k` is the rate constant for reaction :math:`r` with reactants :math:`q`. We add this amount to each
product of the reaction step, whereas we subtract it from each reactant.

.. _Jacobian:

Estimating the Jacobian
-----------------------
The Jacobian matrix is a matrix containing the partial derivatives of the rate equations of chemical :math:`i` with
respect to each chemical :math:`j`. Similarly to how we analyzed the rate equations we can again decompose the entire
model into individual reaction steps:

.. math::

    J_{i, j} =  \frac{\delta (di/dt)}{\delta j} = \sum_{r}{\frac{\delta (di_r/dt)}{\delta j}}

We can again initialize a matrix containing only zeros, loop over each reaction step, and calculate the partial
derivative with respect to each reactant. Partial derivatives with respect to a product do not have to be considered
as the corresponding rate equation would not contain a term including it and therefore be zero
(:math:`d(k \cdot a \cdot b)/dc=0`, whereas :math:`d(k \cdot a \cdot b)/db=k \cdot a`).

To calculate the partial derivative we than take the product of the concentrations of all reactants, :math:`q`,
except the reactant whose derivative we take.

.. math::

    \frac{\delta (di_r/dt)}{\delta j} = k_r \cdot \prod^{q_r}_{q_r \ne j}{[q_r]}

Subsequently we multiply this with the rate constant, :math:`k`, and add this to all reaction products, whereas
we subtract it from each reactant. Because we take a very simple approach to calculating the derivative, this method only
works for reaction steps which are first order in each chemical.

Explict Euler formula
---------------------
The explicit Euler formula takes the rate of changes as calculated above, and adds it to the currently known concentrations.
It repeats this the number of ``steps_per_step`` times, and discards the intermediate results. The last
array of predicted concentration is saved at the corresponding time stamp.

This method does not work well in stiff problems and using an ODE solver is recommended.


