Predict
=======
The predict module implements the :class:`DRL <predict.DRL>` class which helps to create a prediction of a chemical
system. When the prediction fails it raises a :exc:`InvalidPredictionError <predict.InvalidPredictionError>`. The DRL
class implements methods to:

#. :meth:`Predict the concentrations<predict.DRL.predict_concentration>` for a DRL experiment using the ODE solver (preferred).
#. Calculate the :ref:`change<rate_equations>`  in chemical concentration as a function of the current concentrations.
#. Calculate the :ref:`Jacobian matrix<Jacobian>` , which is required by the ODE solver. This **only** works for reaction where each reaction step is first order in each chemical.
#. Predict the concentrations for a DRL experiment using the explicit Euler formula (discouraged).

.. py:currentmodule:: predict
.. class:: DRL(reactions, rate_constants, output_order=None, verbose=False)

    :param reactions: The reactions that describe the system. Each tuple exist out of the name of the rate constant,
        a list of reactants, and a list of products.
    :type reactions: list[tuple[str, list[str], list[str]]]
    :param rate_constants: A dictionary which maps the rate constants to their respective values.
    :type rate_constants: dict[str: float]
    :param output_order: Defines in which column the concentration of each chemical will be stored.
            By default, it is alphabetical.
    :type output_order: list[str]
    :param verbose: If True, it will print and store information on which reactions are initialized.
    :type verbose: bool

    :ivar reactions_overview: A comprehensive overview of the reactions that will be calculated
        if the verbose argument was True upon initialization.

    .. method:: predict_concentration(t_eval_pre, t_eval_post, initial_concentrations, labeled_concentrations, dilution_factor, atol=1e-10, rtol=1e-10)

        Predicts the concentrations during a DRL experiment. After evaluating all timestamp in t_eval_pre it 'dilutes'
        the prediction and sets the concentration of labeled compound. Subsequently the t_eval_post time stamps are
        evaluated. It utilizes the ODE solver 'scipy.integrate.solve_ivp' with the Radau method. Negative concentrations
        might occur within the range of the tolerances given.

        :param t_eval_pre: The time steps that must be evaluated and returned before the addition of the labeled compound.
        :param t_eval_post:  The time steps that must be evaluated and returned after the addition of the labeled compound.
        :param initial_concentrations: The initial concentrations of each chemical. Non-zero concentrations are not required.
        :param labeled_concentration: The concentration of the labeled chemical. Non-zero concentrations are not required. This concentration is not diluted.
        :param dilution_factor: The factor (≤1) by which the prediction will be multiplied when the labeled chemical is added.
            This simulates the dilution upon addition of solvent.
        :param atol: The absolute tolerances for the ODE solver.
        :param rtol: The relative tolerances for the ODE solver.

        :type t_eval_pre: np.ndarray
        :type t_eval_post:  np.ndarray
        :type initial_concentrations: dict[str: float]
        :type labeled_concentration: dict[str: float]
        :type dilution_factor: float
        :type atol: float
        :type rtol: float

        :raise InvalidPredictionError: If negative concentration larger than the maximum tolerance, or NaN values, are detected in the output.
        :return: Predictions of the concentrations pre-, and post-addition of the labeled compound.
        :rtype: tuple[polars.DataFrame, polars.DataFrame]

    .. method:: predict_concentration_Euler(experimental_conditions, steps_per_step=1)

        Predicts the concentrations during a DRL experiment. After evaluating all timestamp in t_eval_pre it 'dilutes'
        the prediction and sets the concentration of labeled compound. Subsequently the t_eval_post time stamps are
        evaluated. It implements the explicit Euler formula. More steps in between each evaluated point in the t_eval
        arrays can be added to increase the accuracy.

        :param t_eval_pre: The time steps that must be evaluated and returned before the addition of the labeled compound.
        :param t_eval_post:  The time steps that must be evaluated and returned after the addition of the labeled compound.
        :param initial_concentrations: The initial concentrations of each chemical. Non-zero concentrations are not required.
        :param labeled_concentration: The concentration of the labeled chemical. Non-zero concentrations are not required. This concentration is not diluted.
        :param dilution_factor: The factor (≤1) by which the prediction will be multiplied when the labeled chemical is added.
            This simulates the dilution upon addition of solvent.
        :param steps_per_step: The number of steps to simulate per step of the t_eval array. Higher values yield higher
            accuracy at the cost of computation time.

        :type t_eval_pre: np.ndarray
        :type t_eval_post:  np.ndarray
        :type initial_concentrations: dict[str: float]
        :type labeled_concentration: dict[str: float]
        :type dilution_factor: float
        :type steps_per_step: int

        :raise InvalidPredictionError: If negative concentration larger than the maximum tolerance, or NaN values, are detected in the output.
        :return: Predictions of the concentrations pre-, and post-addition of the labeled compound.
        :rtype: tuple[polars.DataFrame, polars.DataFrame]


.. exception:: InvalidPredictionError

    Raised when NaN values or values more negative than the tolerance are found.
    For debugging purposes the error also contains the rate constants which were used when the error occurred.


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

We can create a prediction using the :func:`DRL.predict_concentration`. The class implements the method which determines
the rate of change as a function of its current state, and a method which calculate the Jacobian based on its state.
Because we do not want to model an entire DRL experiment, ``solve_ivp`` is used instead of :func:`DRL.predict_concentration`.
Internally, the function calls ``solve_ivp``.

.. code-block:: python

    import numpy as np
    from scipy.integrate import solve_ivp
    from delayed_reactant_labeling.predict import DRL

    drl = DRL(rate_constants=rate_constants, reactions=reactions, output_order=['A', 'B', 'C'], verbose=False)
    result = solve_ivp(drl.calculate_step, t_span=[0, 20], y0=[A0, 0, 0], method='Radau', t_eval=time, jac=drl.calculate_jac)

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

    ax.plot(time, A0 * np.exp(-k1 * time), color='k', linestyle=':', label='algebraic')
    ax.plot(time, k1 / (k2 - k1) * A0 * (np.exp(-k1 * time) - np.exp(-k2 * time)), color='k', linestyle=':')
    ax.plot(time, A0 * (1 - np.exp(-k1 * time) - k1 / (k2 - k1) * (np.exp(-k1 * time) - np.exp(-k2 * time))), color='k', linestyle=':')
    ax.legend()
    fig.show()

.. image:: images/predict_prediction.png
    :width: 600
    :align: center

It is clear that the model fits the data very well, and its much easier to implement these few lines of code, instead of
doing the mathematics. Furthermore, implementing more difficult problems only requires the addition of a few lines here,
whereas solving the problem in an exact manner becomes impossible.
