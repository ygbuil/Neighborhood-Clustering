# libraries
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns


def exploratory_data_analysis(c, df):
    '''
    Check data distribution, missing values, duplicates and perform corrective
    tasks as well as extract plots.

    Parameters
    ----------
    c : instance of class
        Instance of calss Constants that contains all constants.
    df : pandas dataframe
        Dataframe to analyze.

    Returns
    -------
    df : pandas dataframe
        Cleaned dataframe.

    '''

    # check data distribution for numerical columns
    print(df.describe())

    # check data types and missing values per column
    print(df.info())

    # remove duplicate rows if exist (based on primary_key = label_column)
    if len(df[df.duplicated(subset=c.label_column)]) > 0:
        df = df[~df.duplicated(subset=df.label_column)]

    # plot correlation matrix
    corr = df.corr()
    sns.heatmap(
        data=corr, xticklabels=corr.columns, yticklabels=corr.columns,
        annot=True
    )
    plt.show()

    # plot relationship between variables
    sns.pairplot(df.drop(c.label_column, axis=1))
    plt.show()

    # plot histogram and boxplot for every column
    columns = list(df.columns)
    columns.remove(c.label_column)
    for column in columns:
        # plot histogram
        plt.hist(x=df[column], bins=20)
        plt.xlabel(column)
        plt.ylabel('frequency')
        plt.show()

        # plot boxplot
        plt.boxplot(df[column], vert=False)
        plt.xlabel(column)
        plt.show()

    return df


def generate_k_evaluation_plots(c, train, kmeans_hyperparams):
    '''
    Calculates "Within Cluster Sum of Squares" and "Silhouette Coefficient" for
    various K values and plots the evolution of both metrics.

    Parameters
    ----------
    c : instance of class
        Instance of calss Constants that contains all constants.
    train : pandas dataframe
        Train data.
    kmeans_hyperparams : dict
        Kmeans hyper-parameters.

    Returns
    -------
    None.

    '''

    wcss = []
    silhouette_coefficients = []

    for k in c.k_values_to_try:
        print(f'Evaluating K = {k}')
        model = KMeans(n_clusters=k, **kmeans_hyperparams)
        model.fit(train)
        wcss.append(model.inertia_)
        silhouette_coefficients.append(silhouette_score(train, model.labels_))

    plot_lineplot(
        x=c.k_values_to_try, y=wcss, x_label='K value', y_label='WCSS'
    )
    plot_lineplot(
        x=c.k_values_to_try, y=silhouette_coefficients, x_label='K value',
        y_label='Silhouette Coefficients'
    )


def plot_lineplot(x, y, x_label, y_label):
    '''
    Plots a lineplot.

    Parameters
    ----------
    x : list
        x values.
    y : list
        y values.
    x_label : str
        x axis label.
    y_label : str
        y axis label.

    Returns
    -------
    None.

    '''

    plt.plot(x, y)
    plt.xticks(x)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()


def format_results(c, df, labels, predicted_clusters):
    '''
    Unifies the original dataframe and the cluster labels predicted by the
    model.

    Parameters
    ----------
    c : instance of class
        Instance of calss Constants that contains all constants.
    df : pandas dataframe
        Original dataframe.
    labels : list
        List of labels used in training/testing.
    predicted_clusters : list
        Predicted cluster labels.

    Returns
    -------
    results : pandas dataframe
        Original dataframe with cluster labels.

    '''

    results = pd.DataFrame(
        data={c.label_column: labels, 'cluster': predicted_clusters}
    )
    results = pd.merge(left=results, right=df, on=c.label_column, how='left')

    return results


def get_cluster_centers(model, train_columns):
    '''
    Gets the cluster center points for the train data.

    Parameters
    ----------
    model : model
        K-means model.
    train_columns : list
        Column names of the train data.

    Returns
    -------
    cluster_centers : pandas dataframe
        Cluster center points.

    '''

    cluster_centers = model.cluster_centers_
    cluster_centers = pd.DataFrame(data=cluster_centers, columns=train_columns)
    cluster_centers.reset_index(inplace=True)
    cluster_centers = cluster_centers.rename(columns={'index': 'cluster'})

    return cluster_centers


def plot_parallel_coordinates(df, plot_title):
    '''
    Plots the values of every observation and the cluster each one belongs to
    using a parallel coordinates plot.

    Parameters
    ----------
    df : pandas dataframe
        Dataframe used to fit the model + cluster results column.
    plot_title : string
        Plot title name.

    Returns
    -------
    None.

    '''

    df = df.sort_values(by='cluster', ascending=True)
    pd.plotting.parallel_coordinates(
        df, class_column='cluster', colormap='viridis'
    )
    plt.title(plot_title)
    plt.show()


def plot_pairplot(df, plot_title):
    '''
    Plots a pairplot with each observation colored according to the cluster it
    belongs.

    Parameters
    ----------
    df : pandas dataframe
        Dataframe to plot.
    plot_title : string
        Plot title name.

    Returns
    -------
    None.

    '''

    plot = sns.pairplot(df, hue='cluster', palette='viridis')
    plot.fig.subplots_adjust(top=0.95)
    plot.fig.suptitle(plot_title)
    plt.show()
