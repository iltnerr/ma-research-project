import os
import pickle

from sql_utilities import ROOT_PATH


def pickle_dump(attribute, filename):
    """Save pickle to directory.
    """

    pickle_dir = os.path.join(ROOT_PATH, 'pickling')

    # Check/Create directory for results
    try:
        if not os.path.isdir(pickle_dir):
            os.makedirs(pickle_dir)
        directory = os.path.join(pickle_dir, filename)
        pickle.dump(attribute, open(directory, "wb"))

    except Exception as e:
        print('Error in pickle_dump: {}'.format(e))


def pickle_load(filename):
    """Load pickle from directory.
    """

    pickle_dir = os.path.join(ROOT_PATH, 'pickling')
    file_dir = os.path.join(pickle_dir, filename)

    attr = None
    try:
        if os.path.isfile(file_dir):
            attr = pickle.load(open(file_dir, "rb"))
        else:
            print("Could not find file to pickle from.")
    except Exception as e:
        print('Error in pickle_load: {}'.format(e))

    return attr

def dataset_by_ids(df, list_of_ids):
    """
    Filter Dataset by ids.

    Returns pd.DataFrame with specified profile ids.
    """

    for id in list_of_ids:
        assert id in set(df['profile_id']), 'profile_id {} is missing in the dataset.'.format(id)

    df_by_ids = df[df.profile_id.isin(list_of_ids)]
    return df_by_ids