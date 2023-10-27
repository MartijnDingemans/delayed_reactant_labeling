Optimize
========

To optimize the rate constants, several steps need to be performed, such as:

1. Defining the error functions
2. Creating a prediction
3. Comparing the result fo the prediction with experimental data
4. Weighing of the errors
5. Saving the intermediate results
6. Repeating the 3-5 until a the error no longer improves and the rate constants have converged to a stable value or the maximum number of iterations has been reached.

The :class:`optimize.RateConstantOptimizerTemplate` is an abstract base class which has implemented most of the steps
above. However, the user must still define its error functions and the exact methodology of creating a prediction.

.. py:currentmodule:: optimize
.. class:: RateConstantOptimizerTemplate(weights, experimental, metric)

    :param weights: A dictionairy containing patterns and weight. Each pattern will be searched for in  the errors.
        Upon a match the error yielded will be multiplied by its respective weight. If a error is matched with multiple
        patterns, the weight will be decreased in a multiplicative manner. The final weights after the pattern search
        are stored in weights_array
    :type weights: dict[str, float]
    :param experimental: The experimental data.
    :type experimental: polars.DataFrame
    :param metric: An error metric which takes as input the keywords ``y_pred`` and ``y_true`` and returns a float. Lower values
        should mean a better prediction.
    :type metric: Callable[[np.ndarray, np.ndarray], float]

    :var weights_array: *(np.ndarray)* - The weights for each error function.

    .. method::`create_prediction(x, x_description)`

        **Must be implemented by the user!** Takes a set of parameters and their respective description, and should
        output the prediction starting at the time that the labeled compound has been added.

        :param x: The parameters required for the reaction.
        :type x: np.ndarray
        :param x_description: The description of each parameter.
        :type x_description: list[str]




