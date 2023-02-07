# libraries
from dataclasses import dataclass


@dataclass
class Constants:
    '''
    Declares all the constants that will be using during the entire code.

    Parameters
    ----------
    label_column : str
        Column name of the label of each observation.
    test_size : float
        Percentage of data used for testing.
    k_value : int
        Definitive K value to be used for the K-means algorithm.
    k_values_to_try : list
        List of K values to try in order to find out the most optimal.
    kmeans_hyperparams : dict
        Hyper-parameters of the K-means algorithm.


    Returns
    -------
    None.

    '''

    label_column: str
    test_size: float
    k_value: int
    k_values_to_try: list
    kmeans_hyperparams: dict
