import pandas as pd
import os
import math


def splite(data_path):
    """
    Splite data in csv file into training set and testing set.

    Parameters:
        data_path: Path to one csv file.
    """
    # read input csv file
    df = pd.read_csv(data_path)
    # calculate length of training set
    training_set_length = math.floor(len(df) * 0.7)
    # training and testing dataframe
    df_training = df.iloc[: training_set_length, :]
    df_testing = df.iloc[training_set_length :, :]
    # save training and testing dataframe to csv file
    base_path = os.path.dirname(__file__)
    df_training.to_csv(os.path.join(base_path, '../../data/splited/train.csv'), index=False)
    df_testing.to_csv(os.path.join(base_path, '../../data/splited/test.csv'), index=False)


if __name__ == "__main__":
    base_path = os.path.dirname(__file__)
    splite(os.path.join(base_path, '../../data/original/train.csv'))