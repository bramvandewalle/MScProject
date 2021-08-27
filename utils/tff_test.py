import pandas as pd
import tensorflow as tf
import tensorflow_federated as tff
from timeit import default_timer as timer

from sklearn.model_selection import train_test_split

from utils import cidds_001 as utils


class TffClientDataProvider:

    def __init__(self, week1, week2, drop_target1, drop_target2, alpha_target1, alpha_target2, normalization_fn, random_state=None):
        self.week1 = week1
        self.week2 = week2
        self.drop_target1 = drop_target1
        self.drop_target2 = drop_target2
        self.alpha_target1 = alpha_target1
        self.alpha_target2 = alpha_target2
        self.random_state = random_state
        self.normalization_fn = normalization_fn

        self._preprocess_datasets()

    def make_client_data(self):
        client_data_dict = {
            'client_1': tf.data.Dataset.from_tensor_slices((self.x_train_week1, self.y_train_week1)),
            'client_2': tf.data.Dataset.from_tensor_slices((self.x_train_week2, self.y_train_week2))
        }

        client_data = tff.simulation.ClientData.from_clients_and_fn(
            client_ids=['client_1', 'client_2'],
            create_tf_dataset_for_client_fn=lambda key: client_data_dict[key]
        )

        return client_data

    def _preprocess_datasets(self):
        print('Start preprocessing datasets week1 and week2')
        start = timer()

       # Balance the week1_prep and week2_prep datasets (ignoring the few entries of the excluded attack type)
        print(f'{round(timer() - start, 2)}s: Creating balanced dataset of week1')
        self.week1_balanced = utils.get_balanced_cidds(self.week1, idx_min_n_after_argsort=1)
        print(f'{round(timer() - start, 2)}s: Creating balanced dataset of week2')
        self.week2_balanced = utils.get_balanced_cidds(self.week2, idx_min_n_after_argsort=1)

        # Normalize the datasets
        print(f'{round(timer() - start, 2)}s: Normalizing week1 and week2')
        self.week1_norm = pd.DataFrame(self.week1_balanced, copy=True)
        self.week2_norm = pd.DataFrame(self.week2_balanced, copy=True)
        
        _ = self.normalization_fn(self.week1_norm, columns_to_normalize=utils.columns_to_normalize)
        _ = self.normalization_fn(self.week2_norm, columns_to_normalize=utils.columns_to_normalize)

        # Remove all but 1-self.alpha_target1 (fraction) of the flows of self.drop_target1 from week1 dataset
        print(f'{round(timer() - start, 2)}s: Removing {100 - 100*self.alpha_target1}% of {self.drop_target1} flows from week1')
        self.week1_balanced = self._remove_attack_type(self.week1_norm, self.drop_target1, self.alpha_target1)

        # Remove all but 1-self.alpha_target2 (fraction) of the flows of self.drop_target2 from week2 dataset
        print(f'{round(timer() - start, 2)}s: Removing {100 - 100*self.alpha_target2}% of {self.drop_target2} flows from week2')
        self.week2_balanced = self._remove_attack_type(self.week2_norm, self.drop_target2, self.alpha_target2)

        # Split datasets in features and one hot encoded labels
        print(f'{round(timer() - start, 2)}s: Separate week1 features from dataset labels and one hot encode the labels')
        self.week1_balanced_x = self.week1_balanced.drop(columns=utils.columns_to_drop + ['attack_type'])
        self.week1_balanced_y = pd.get_dummies(self.week1_balanced['attack_type'])
        self.ohe_columns = self.week1_balanced_y.columns # save the order of the ohe columns

        print(f'{round(timer() - start, 2)}s: Separate week2 features from dataset labels and one hot encode the labels')
        self.week2_balanced_x = self.week2_balanced.drop(columns=utils.columns_to_drop + ['attack_type'])
        self.week2_balanced_y = pd.get_dummies(self.week2_balanced['attack_type'])

        # Split balanced datasets in training and testing sets
        print(f'{round(timer() - start, 2)}s: Split datasets in training and testing datasets')
        self._train_test_split()

        # Convert training and testing sets to numpy arrays
        print(f'{round(timer() - start, 2)}s: Convert features and labels to numpy arrays')
        self._convert_to_numpy()

        print(f'{round(timer() - start, 2)}s: Finished preprocessing datasets week1 and week2')

    def _remove_attack_type(self, dataset, target, alpha):
        # print(f'len(dataset) = {len(dataset)}, target = {target}, alpha = {alpha}')
        prep = dataset.where(dataset['attack_type'] != target).dropna()
        # print(f'len(prep) = {len(prep)}')
        targ = dataset.where(dataset['attack_type'] == target).dropna()
        # print(f'len(targ) = {len(targ)}')
        # print(f'int(alpha*len(targ)) = {int(alpha*len(targ))}')
        targ = targ.head(n=int(alpha*len(targ)))
        prep = prep.append(targ).sample(frac=1, random_state=self.random_state)
        # print(f'len(prep) = {len(prep)}')
        return prep

    def _train_test_split(self, test_size=0.2):
        self.x_train_week1, self.x_test_week1, self.y_train_week1, self.y_test_week1 = train_test_split(
            self.week1_balanced_x, self.week1_balanced_y, test_size=test_size, random_state=self.random_state)

        self.x_train_week2, self.x_test_week2, self.y_train_week2, self.y_test_week2 = train_test_split(
            self.week2_balanced_x, self.week2_balanced_y, test_size=test_size, random_state=self.random_state)

    def _convert_to_numpy(self):
        # week 1
        self.x_train_week1 = self.x_train_week1.to_numpy()
        self.x_test_week1  = self.x_test_week1.to_numpy()
        self.y_train_week1 = self.y_train_week1.to_numpy()
        self.y_test_week1  = self.y_test_week1.to_numpy()

        # week 2
        self.x_train_week2 = self.x_train_week2.to_numpy()
        self.x_test_week2  = self.x_test_week2.to_numpy()
        self.y_train_week2 = self.y_train_week2.to_numpy()
        self.y_test_week2  = self.y_test_week2.to_numpy()
