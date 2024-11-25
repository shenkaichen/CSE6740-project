import os
import pandas as pd


if __name__ == "__main__":
    
    # read training and testing data
    base_path = os.path.dirname(__file__)
    train = pd.read_csv(os.path.join(base_path, '../../data/splited/train.csv'))
    test = pd.read_csv(os.path.join(base_path, '../../data/splited/test.csv'))
    train_label = train.iloc[:, 1]
    train_features = train.iloc[:, 2 :]
    test_label = test.iloc[:, 1]
    test_features = test.iloc[:, 2 :]