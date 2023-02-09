# libraries
import os
import sys
from pathlib import Path

# root path
path = Path('C:/Users/llorenc.buil/github/Neighborhood-Clustering')
os.chdir(path)
if path not in sys.path:
    sys.path.append(path)

# local libraries
from src.constants.constants import c
from src.objects import main_modules as m


def main(c):
    '''
    Entire pipeline. Plots conclusions and returns dataframes with results.

    Parameters
    ----------
    c : instance of class
        Instance of calss Constants that contains all constants.

    Returns
    -------
    results_train : pandas dataframe
        Dataframe containing labels + clusters + attributes (with data used
        during training).
    results_test : pandas dataframe
        Dataframe containing labels + clusters + attributes (with data used
        during testing).
    cluster_centers : pandas dataframe
        Dataframe containing labels + clusters + attributes of the cluster
        centers obtained during training.

    '''

    # read inputs
    df = m.read_inputs(c=c, file_path=Path('data/neighborhood_info.csv'))

    # preprocessing
    train, test, labels_train, labels_test, train_columns = (
        m.preprocessing(c=c, df=df)
    )

    # train
    model = m.train(c=c, train=train)

    # get results
    results_train, results_test, cluster_centers = m.get_results(
        c=c, df=df, model=model, train=train, test=test,
        labels_train=labels_train, labels_test=labels_test,
        train_columns=train_columns
    )

    return results_train, results_test, cluster_centers


if __name__ == '__main__':
    results_train, results_test, cluster_centers = main(c=c)
