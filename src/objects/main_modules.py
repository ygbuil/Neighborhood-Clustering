# libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# local libraries
from src.objects import functions as f


def read_inputs(c, file_path):
    '''
    Read input data.

    Parameters
    ----------
    c : instance of class
        Instance of calss Constants that contains all constants.
    file_path : string
        Demographic data csv file path.
    Returns
    -------
    df : pandas dataframe
        Dataframe with demographic data.

    '''

    df = pd.read_csv(file_path)
    df[c.label_column] = df[c.label_column].astype(str)

    return df


def preprocessing(c, df):
    '''
    Perform exploratory data analysis, train-test split and data
    standarization.

    Parameters
    ----------
    c : instance of class
        Instance of calss Constants that contains all constants.
    df : pandas dataframe
        Dataframe to preprocess.

    Returns
    -------
    train : pandas dataframe
        Train dataframe.
    test : pandas dataframe
        Test dataframe.
    labels_train : pandas dataframe
        Train labels.
    labels_test : pandas dataframe
        Test labels.
    train_columns : list
        Train column names.

    '''

    # exploratory data analysis
    df = f.exploratory_data_analysis(c=c, df=df)

    # train test split
    train, test = train_test_split(df, test_size=c.test_size)
    train, test, labels_train, labels_test = (
        train.drop(c.label_column, axis=1).reset_index(drop=True),
        test.drop(c.label_column, axis=1).reset_index(drop=True),
        train[c.label_column].reset_index(drop=True),
        test[c.label_column].reset_index(drop=True)
    )
    train_columns = train.columns

    # standarize data
    sc = StandardScaler()
    train = pd.DataFrame(data=sc.fit_transform(train), columns=train.columns)
    test = pd.DataFrame(data=sc.fit_transform(test), columns=test.columns)

    return train, test, labels_train, labels_test, train_columns


def train(c, train):
    '''
    Train K-means for multiple K values and use the most optimal K value for
    the final model.

    Parameters
    ----------
    c : instance of class
        Instance of calss Constants that contains all constants.
    train : pandas dataframe
        Train dataframe.

    Returns
    -------
    model : sklearn model
        Trained K-means model.

    '''

    # generate elbow plot to chose best K
    f.generate_k_evaluation_plots(
        c=c, train=train, kmeans_hyperparams=c.kmeans_hyperparams
    )

    # train with the best K value observed on the previously plotted graphs
    model = KMeans(n_clusters=c.k_value, **c.kmeans_hyperparams)
    model.fit(train)

    return model


def get_results(
    c, df, model, train, test, labels_train, labels_test, train_columns
):
    '''
    Get the cluster results for train, test and cluster centers in various
    dataframes as well as plots.

    Parameters
    ----------
    c : instance of class
        Instance of calss Constants that contains all constants.
    df : pandas dataframe
        Original/initial ataframe.
    model : sklearn model
        Trained K-means model.
    train : pandas dataframe
        Train dataframe.
    test : pandas dataframe
        Test dataframe.
    labels_train : pandas dataframe
        Train labels.
    labels_test : pandas dataframe
        Test labels.
    train_columns : list
        Train column names.

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

    # get results train
    prediction_train = model.labels_
    results_train = f.format_results(
        c=c, df=df, labels=labels_train, predicted_clusters=prediction_train
    )

    # get results test (to check for overfitting)
    prediction_test = model.predict(test)
    results_test = f.format_results(
        c=c, df=df, labels=labels_test, predicted_clusters=prediction_test
    )

    # get cluster centers
    cluster_centers = f.get_cluster_centers(
        model=model, train_columns=train_columns
    )

    # plot train cluster results
    train['cluster'] = prediction_train
    f.plot_parallel_coordinates(df=train, plot_title='Train clusters')
    f.plot_pairplot(
        df=results_train.drop(c.label_column, axis=1),
        plot_title='Train clusters'
    )

    # plot test cluster results
    test['cluster'] = prediction_test
    f.plot_parallel_coordinates(df=test, plot_title='Test clusters')
    f.plot_pairplot(
        df=results_test.drop(c.label_column, axis=1),
        plot_title='Test clusters'
    )

    # plot center results
    f.plot_parallel_coordinates(
        df=cluster_centers, plot_title='Center clusters'
    )

    return results_train, results_test, cluster_centers
