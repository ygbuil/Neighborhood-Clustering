# local libraries
from src.objects.constants import Constants


c = Constants(
    label_column='postcode',
    test_size=0.1,
    k_value=4,
    k_values_to_try=range(2, 11),
    kmeans_hyperparams={
        'init': 'random', 'n_init': 10, 'max_iter': 300, 'random_state': 42
    }
)
