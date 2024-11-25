import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score


if __name__ == "__main__":
    
    # read training and testing data
    base_path = os.path.dirname(__file__)
    train = pd.read_csv(os.path.join(base_path, '../../data/splited/train.csv'))
    test = pd.read_csv(os.path.join(base_path, '../../data/splited/test.csv'))
    train_label = train.iloc[:, 1]
    train_features = train.iloc[:, 2 :]
    test_label = test.iloc[:, 1]
    test_features = test.iloc[:, 2 :]

    # standardize the training and testing features
    scaler = StandardScaler()
    scaler.fit(train_features)
    train_features_standard = scaler.transform(train_features)
    test_features_standard = scaler.transform(test_features)

    # train logistic regression model on the training set
    lr_model = LogisticRegression(penalty='none', max_iter=50)
    lr_model.fit(train_features_standard, train_label)

    # evaluate the trained model on the testing set based on AUC
    test_label_pred = lr_model.predict(test_features_standard)
    auc_score = roc_auc_score(test_label, test_label_pred)
    print(f"AUC: {auc_score}")