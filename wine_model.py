from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

d = load_wine()
print(d['DESCR'])
X = pd.DataFrame(d['data'], columns=d['feature_names'])
y = d['target']  # cultivator

def train_model(X,y):
    m = RandomForestClassifier()
    m.fit(X,y)
    return m