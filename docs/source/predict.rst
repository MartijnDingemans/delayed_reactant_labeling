Predict
=======
The predict module implements the :class:`DRL <predict.DRL>` class which helps to create a prediction of a chemical
system. When the prediction fails it raises a :exc:`InvalidPredictionError <predict.InvalidPredictionError>`. The prediction requires the class
:class:`experimental data <predict.Experimental_Conditions>` to contain all relevant data.


.. py:currentmodule:: predict
.. class:: DRL(reactions, rate_constants, verbose=False)

    Upon creation of the DRL class object, it will convert the reaction equations to their respective rate equations.
    When the model is given a set of initial concentrations, the rate equations can used to predict the concentration of
    the system for the next time stamp (explicit Euler formula).

    :param reactions: The reactions that describe the system. Each tuple exist out of the name of the rate constant,
        a list of reactants, and a list of products.
    :type reactions: list[tuple[str, list[str], list[str]]]
    :param rate_constants: A dictionary which maps the rate constants to their respective values.
    :type rate_constants: dict[str: float]
    :param verbose: If True, it will print and store information on which reactions are initialized.
    :type verbose: bool


    :ivar reactions_overview: A comprehensive overview of the reactions that will be calculated
        if the verbose argument was True upon initialization.

    .. method:: predict_concentration(experimental_conditions, steps_per_step=1)

        Calls self._predict_concentration_slice to predict both the time before the addition, as the situation after the
        addition of the labeled compound. Inbetween the two prediction the concentration will be diluted according to
        the dilution factor, and the labeled chemical will be introduced in the system.

        :param experimental_conditions: The experimental conditions for which the prediction should be made.
        :type experimental_conditions: :class:`Experimental_Conditions`
        :param steps_per_step: The number of steps inbetween each step in the experimental_conditions.time array.
        :type steps_per_step: int
        :raise InvalidPredictionError: If negative concentration, or NaN values, are detected in the output.
        :return: Predictions of the concentrations pre-, and post-addition of the labeled compound.
            The chemicals might be in ordered in a different position.
        :rtype: tuple[polars.DataFrame, polars.DataFrame]


.. exception:: InvalidPredictionError

    Raised if negative or NaN values are detected in the output of the prediction. Negative values can occur, when a the
    rate constant multiplied with the delta time and the concentration of reactants becomes larger than the concentration
    of the reactant. By decreasing the delta time, we can prevent this from happening. One convenient way of doing this
    is by increasing the ``steps_per_step`` of the measured data points.

    For debugging purposes the error also contains the rate constants which were used when the error occurred.


.. class:: Experimental_Conditions(time, initial_concentrations, dilution_factor, labeled_reactant)

    Stores basic information required for a prediction.

    :param time: The time points at which a mass spectrum was measured, for both the unlabeled and the labeled sections.
    :type time: tuple[np.ndarray, np.ndarray]
    :param initial_concentrations: The initial concentrations for each chemical. Only non-zero values must be given.
    :type initial_concentrations: dict[str, float]
    :param dilution_factor: The factor (< 1) by which all concentration will be multiplied upon the addition of the labeled reactant.
    :type dilution_factor: float
    :param labeled_reactant: The initial concentration for the labeled reactant. This will **not** be diluted.


example
-------
The simple chemical system:

.. math::
    A \xrightarrow{\text{k1}} B \xrightarrow{\text{k2}} C

can be modeled using the :class:`DRL <predict.DRL>` class. First the reaction scheme should be written in a code
friendly way:

.. code-block:: python

    reaction1 = ('k1', ['A'], ['B'])
    reaction2 = ('k2', ['B'], ['C'])
    reactions = [reaction1, reaction2]

Where the first element of each tuple is the name of the corresponding rate constant, the second element is a list
containing all reactants, and the third element is a list containing all the products. If for example B split into C and
byproduct D, we could write the reaction2 as ``reaction2 = ('k2', ['B'], ['C', 'D'])``

Lets assume that we know the rate constants belonging to this reaction.

.. code-block:: python

    rate_constants = {
        "k1": 0.2,
        "k2": 0.5,
    }

To create the relevant prediction for this model we need to collect the
:class:`experimental data <predict.Experimental_Conditions>` in a class. Some of the input variables contain non-sensible
information, this is because we currently do not want to model a complete DRL experiment, but only the simple beginning.
After this is done, the prediction can be made using :func:`DRL.predict_concentration`.

.. code-block:: python

    import numpy as np
    from delayed_reactant_labeling.predict import DRL, Experimental_Conditions

    time_pre_addition = np.linspace(0, 20, 2000)  # 0 to 10 minutes, 2000 datapoints
    time_post_addition = np.linspace(0, 1, 10)    # filled with a useable time array, but we will ignore these results
    A0 = 1

    experimental_conditions = Experimental_Conditions(
        time=(time_pre_addition, time_post_addition,),
        initial_concentrations={'A': A0},
        dilution_factor=1,                  # we do not model DRL here
        labeled_reactant={}                 # we do not model DRL here
    )

    drl = DRL(rate_constants=rate_constants, reactions=reactions, verbose=False)
    pred, _ = drl.predict_concentration(experimental_conditions=experimental_conditions)

However, also algebraic `solutions <https://chem.libretexts.org/Bookshelves/Physical_and_Theoretical_Chemistry_Textbook_Maps/Mathematical_Methods_in_Chemistry_(Levitus)/04%3A_First_Order_Ordinary_Differential_Equations/4.03%3A_Chemical_Kinetics>`_
for this specific chemical problem exist.

.. math::
   :nowrap:

    \begin{eqnarray}
    [A]_t = [A]_0 \cdot e^{-k_1t}
    \end{eqnarray}
    \begin{eqnarray}
    [B]_t = \frac{k_1}{k_2-k_1}[A]_0(e^{-k_1t}-e^{-k_2t})
    \end{eqnarray}
    \begin{eqnarray}
    [C]_t = A_0[1-e^{-k_1t}-\frac{k_1}{k_2-k_1}(e^{-k_1t}-e^{-k_2t})]
    \end{eqnarray}

We can compare the algebraic solution to the modelled prediction as follows.

.. code-block:: python

    import matplotlib.pyplot as plt
    time = experimental_conditions.time[0]
    k1, k2 = rate_constants['k1'], rate_constants['k2']

    fig, ax = plt.subplots()
    ax.plot(time, pred['A'] / A0, label='A')
    ax.plot(time, pred['B'] / A0, label='B')
    ax.plot(time, pred['C'] / A0, label='C')

    ax.plot(time, A0 * np.exp(-k1 * time),
        color='k', linestyle=':', label='algebraic')
    ax.plot(time, k1 / (k2 - k1) * A0 * (np.exp(-k1 * time) - np.exp(-k2 * time)),
        color='k', linestyle=':')
    ax.plot(time, A0 * (1 - np.exp(-k1 * time) - k1 / (k2 - k1) * (np.exp(-k1 * time) - np.exp(-k2 * time))),
        color='k', linestyle=':')
    ax.legend()
    fig.show()

.. image:: images/predict_prediction.png
    :width: 600
    :align: center

It is clear that the model fits the data very well, and its much easier to implement these few lines of code, instead of
doing the mathematics. Furthermore, implementing more difficult problems only requires the addition of a few lines here,
whereas solving the problem in an exact manner becomes impossible.
