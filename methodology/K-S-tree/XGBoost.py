import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
import os
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

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

dtrain = xgb.DMatrix(train_features_standard, label=train_label)
dtest = xgb.DMatrix(test_features_standard, label=test_label)

params = {
    'objective': 'binary:logistic',
    'max_depth': 4,
    'eta': 0.3,
    'scale_pos_weight': 1,
    'eval_metric': 'auc'
}

bst = xgb.train(params, dtrain, num_boost_round=10)

test_label_pred = bst.predict(dtest)

auc_score = roc_auc_score(test_label, test_label_pred)
print(f"AUC: {auc_score}")

fpr, tpr, thresholds = roc_curve(test_label, test_label_pred)
plt.figure()
plt.plot(fpr, tpr, linewidth=0.8, label='XGBoost')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('ROC Curve')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()