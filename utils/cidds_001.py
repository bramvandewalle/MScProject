import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler


columns_to_drop = [
    'date_first_seen',
    'src_ip_addr',
    'dst_ip_addr',
    'flows',
    'class',
    'attack_id',
    'attack_description'
]

columns_to_normalize = [
    'duration',
    'src_port',
    'dst_port',
    'packets',
    'bytes',
    'tos'
]


def clean_bytes(bytes):
    """
    Auxiliary method that maps the input string representing
    the number of bytes used in a flow to a floating point value.

    The only caveat there is in the CIDDS-001 datasets is that if
    the number of bytes exceeds one million, then the value is
    represented with an 'M'. So '6,000,000' is represented as '6 M'.
    However, we need the numerical representation. This is taken
    care of in this auxiliary method.

    Params
    ------
        - bytes: string, textual representation of number of bytes
    
    Returns
    -------
        - float, numerical representation of number of bytes
    """
    param = str(bytes)
    if ' M' in param:
        param = param.replace(' M', 'e6')
    return float(param)


def clean_cidds_001(data):
    """
    Clean the given CIDDS-001 dataset.

    Most of the columns are left untouched, however the values in column `Bytes`
    are mapped to a numerical representation (see clean_bytes() auxiliary method).
    Also, the columns `Proto` and `Flags` are transformed. `
    
    Proto` is one-hot-encoded in the cleaned dataset and the values of `Flags` are 
    expanded over six columns. The values in `Flags` are 6 characters long strings
    representing whether or not any of the six TCP flags URG, ACK, PSH, RST, SYN 
    and/or FIN were set in the flow of communication. In the cleaned dataset, instead
    of using one string with six characters, six binary columns are used.

    Finally, all column names are renamed to be consistent with one another. I.e.
    all lower case and in case of multiple words an underscore (_) is used inbetween.

    Params
    ------
        - data: pandas.DataFrame, raw CIDDS-001 dataset
    
    Returns:
    --------
        - pandas.DataFrame, cleaned CIDDS-001 dataset
    """
    # get all columns to be used
    data_first_seen = data['Date first seen']
    duration = data['Duration']
    proto = data['Proto']
    src_ip = data['Src IP Addr']
    src_prt = data['Src Pt']
    dst_ip = data['Dst IP Addr']
    dst_pt = data['Dst Pt']
    packets = data['Packets']
    bytes = data['Bytes']
    flows = data['Flows']
    flags = data['Flags']
    tos = data['Tos']
    clss = data['class']
    attackType = data['attackType']
    attackID = data['attackID']
    attackDescription = data['attackDescription']

    # one-hot encoding of prototype
    proto_df = pd.get_dummies(proto)

    # some bytes contain '... M' to represent ... milion, make sure that all values are floats
    bytes = bytes.map(lambda b: clean_bytes(b))
    
    # binary encoding of the 6 TCP flags instead of representing them all in one string
    flags_df = pd.DataFrame(
        data=[[0 if c == '.' else 1 for c in str] for str in flags],
        columns=['tcp_urg', 'tcp_ack', 'tcp_psh', 'tcp_rst', 'tcp_syn', 'tcp_fin']
    )

    # put everything in one DataFrame
    columns = [data_first_seen, duration, proto_df, src_ip, src_prt, dst_ip, dst_pt,
        packets, bytes, flows, flags_df, tos, clss, attackType, attackID, attackDescription]
    dataset = pd.concat(columns, axis=1)

    # rename the columns
    dataset = dataset.rename(str.lower, axis='columns') # lower
    dataset = dataset.rename(str.strip, axis='columns') # remove leading/training whitespaces
    dataset = dataset.rename(lambda str: str.replace('date first seen', 'date_first_seen'), axis='columns')
    dataset = dataset.rename(lambda str: str.replace('src ip addr', 'src_ip_addr'), axis='columns')
    dataset = dataset.rename(lambda str: str.replace('src pt', 'src_port'), axis='columns')
    dataset = dataset.rename(lambda str: str.replace('dst ip addr', 'dst_ip_addr'), axis='columns')
    dataset = dataset.rename(lambda str: str.replace('dst pt', 'dst_port'), axis='columns')
    dataset = dataset.rename(lambda str: str.replace('attacktype', 'attack_type'), axis='columns')
    dataset = dataset.rename(lambda str: str.replace('attackid', 'attack_id'), axis='columns')
    dataset = dataset.rename(lambda str: str.replace('attackdescription', 'attack_description'), axis='columns')

    return dataset


def get_balanced_cidds(cidds_df, classification_target='attack_type', idx_min_n_after_argsort=0):
    '''Creates a balanced dataset.

    Params:
    -------
        - cidds_df (pandas.DataFrame): the dataset to balance
        - classification_target (str): the column in cidds_df that must be balanced
        - idx_min_n_after_argsort (int): this method groups the cidds_df by the classification_target
            and calculates the size of each of the groups. If idx_min_n_after_argsort=0, then the
            smallest group is taken as reference to balance the cidds_df. But the smallest group
            could be too small, so you optionally increase the index to use the second smallest,
            third smallest... group as a reference to balance out the cidds_df dataset.
    '''
    # get all attack types and their corresponding sizes
    temp = cidds_df.groupby(by=classification_target).size()

    # obtain the smallest size
    sizes = []
    for idx in temp.index:
        sizes.append(temp[idx])
    min_n_week1 = np.sort(np.array(sizes))[idx_min_n_after_argsort]

    # create a balanced dataset
    balanced_df = pd.DataFrame()
    for idx in temp.index:
        balanced_df = balanced_df.append(
            other=cidds_df.where(cidds_df[classification_target] == idx).dropna().head(n=min_n_week1),
            ignore_index=True
        )
    
    return balanced_df


def get_dummies(labels):
    res = np.zeros((len(labels), 5))
    dict = {
        '---': 0,
        'bruteForce': 1,
        'dos': 2,
        'pingScan': 3,
        'portScan': 4
    }
    for idx in range(len(labels)):
        res[idx][dict[labels[idx]]] = 1
    return pd.DataFrame(res, columns=['---', 'bruteForce', 'dos', 'pingScan', 'portScan']).astype(int)


def min_max_normalization(cidds_df_train, columns_to_normalize, cidds_df_test=None):
    norm_params = {}

    # calculate the normalization parameters
    for column in columns_to_normalize:
        # obtain min and max of the training data
        min = cidds_df_train[column].min()
        max = cidds_df_train[column].max()
        # and save the normalization parameters
        norm_params[column] = (min, max)
    
    # normalize the train and test sets
    _min_max_normalization_with_given_params(cidds_df_train, columns_to_normalize, norm_params)
    if cidds_df_test is not None:
        _min_max_normalization_with_given_params(cidds_df_test, columns_to_normalize, norm_params)
    
    return norm_params


def _min_max_normalization_with_given_params(cidds_df, columns_to_normalize, norm_params):
    # temporarily dissable changed assignment warning
    pd.options.mode.chained_assignment = None

    for column in columns_to_normalize:
        # obtain min and max from norm_params (normalization parameters)
        min, max = norm_params[column]
        # and perform min-max normalization on cidds_df
        cidds_df[column] = (cidds_df[column] - min) / (max - min)

    # re-enable the changed assignement warning
    pd.options.mode.chained_assignment = 'warn'


def z_score_normalizations_with_given_params(cidds_df, columns_to_normalize, norm_params):
    # temporarily dissable changed assignement warning
    pd.options.mode.chained_assignment = None

    for column in columns_to_normalize:
        # obtain mean and std from norm_params (normalization parameters)
        mean, std = norm_params[column]
        # and perform z-score normalization on cidds_df
        cidds_df[column] = (cidds_df[column] - mean) / std

    # re-enable the changed assignement warning
    pd.options.mode.chained_assignment = 'warn'


def z_score_normalization(cidds_df_train, columns_to_normalize, cidds_df_test=None):
    norm_params = {}

    # calculate the normalization parameters
    for column in columns_to_normalize:
        # obtain mean and std of the training data
        mean = cidds_df_train[column].mean()
        std = cidds_df_train[column].std()
        # and save the normalization parameters
        norm_params[column] = (mean, std)

    # normalize the training and test set
    z_score_normalizations_with_given_params(cidds_df_train, columns_to_normalize, norm_params)
    if cidds_df_test is not None:
        z_score_normalizations_with_given_params(cidds_df_test, columns_to_normalize, norm_params)

    return norm_params


def robust_scaling_with_given_params(cidds_df, columns_to_normalize, norm_params):
    # temporarily dissable changed assignement warning
    pd.options.mode.chained_assignment = None

    for column in columns_to_normalize:
        # obtain median, p25 and p75 from norm_params (normalization parameters)
        median, p25, p75 = norm_params[column]
        # and perform robust scaling on cidds_df
        cidds_df[column] = (cidds_df[column] - median) / (p75 - p25)

    # re-enable the changed assignement warning
    pd.options.mode.chained_assignment = 'warn'


def robust_scaling(cidds_df_train, columns_to_normalize, cidds_df_test=None):
    norm_params = {}

    # calculate the normalization parameters
    for column in columns_to_normalize:
        # obtain median, p25 and p75 of the training data
        median = cidds_df_train[column].median()
        p25 = cidds_df_train[column].quantile(q=0.25)
        p75 = cidds_df_train[column].quantile(q=0.75)
        # and save the normalization parameters
        norm_params[column] = (median, p25, p75)

    # normalize the training set and test set if provided
    robust_scaling_with_given_params(cidds_df_train, columns_to_normalize, norm_params)
    if cidds_df_test is not None:
        robust_scaling_with_given_params(cidds_df_test, columns_to_normalize, norm_params)
    
    return norm_params


def analyze_classification_results(predicted_y, actual_y):
    # obtain results per attack_type
    results = {}

    for idx in range(len(predicted_y)):
        if not actual_y.iloc[idx] in results:
            results[actual_y.iloc[idx]] = [0, 0] # correct, total

        if predicted_y[idx] == actual_y.iloc[idx]:
            results[actual_y.iloc[idx]][0] += 1
        
        results[actual_y.iloc[idx]][1] += 1

    # put results in a dataframe
    results_df = pd.DataFrame([
        [key, results[key][0], results[key][1], results[key][0] / results[key][1]] for key in results
    ], columns=['attack_type', 'correct', 'total', 'acc'])

    # overall results
    corr, tot = 0, 0
    for key in results:
        corr += results[key][0]
        tot += results[key][1]

    results_df = results_df.append(
        pd.DataFrame(
            [['total', corr, tot, corr / tot]],
            columns=['attack_type', 'correct', 'total', 'acc']
        )
    ).reset_index(drop=True)

    return results_df


def load_internal_week1():
    return pd.read_feather('saved_dfs/cidds-001/traffic/OpenStack/CIDDS-001-internal-week1-cleaned.feather')


def load_internal_week2():
    return pd.read_feather('saved_dfs/cidds-001/traffic/OpenStack/CIDDS-001-internal-week2-cleaned.feather')


def load_internal_week3():
    return pd.read_feather('saved_dfs/cidds-001/traffic/OpenStack/CIDDS-001-internal-week3-cleaned.feather')


def load_internal_week4():
    return pd.read_feather('saved_dfs/cidds-001/traffic/OpenStack/CIDDS-001-internal-week4-cleaned.feather')


def load_internal_week1_balanced(random_state=None):
    # Load data
    week1 = pd.read_feather('saved_dfs/cidds-001/traffic/OpenStack/CIDDS-001-internal-week1-cleaned.feather').sample(
        frac=1, random_state=47).reset_index(drop=True)
    
    # Create balanced number of attack flows
    balanced = get_balanced_cidds(week1)

    # Split balanced dataset in train and test datasets
    return _train_test_split_balanced(balanced, random_state=random_state)


def load_internal_week2_balanced(random_state=None):
    # Load data
    week1 = pd.read_feather('saved_dfs/cidds-001/traffic/OpenStack/CIDDS-001-internal-week2-cleaned.feather').sample(
        frac=1, random_state=47).reset_index(drop=True)
    
    # Create balanced number of attack flows
    balanced = get_balanced_cidds(week1)

    # Split balanced dataset in train and test datasets
    return _train_test_split_balanced(balanced, random_state=random_state)


def _train_test_split_balanced(balanced: pd.DataFrame, random_state=None):
    x = balanced.drop(columns='attack_type')
    y = balanced['attack_type']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=random_state)

    return x_train, x_test, y_train, y_test
