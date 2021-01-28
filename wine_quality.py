# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %%
# we need to user regression to predict quality column (3-5)


# %%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
#import mlflow
#import mlflow.sklearn
import logging
from urllib.parse import urlparse
import warnings
warnings.filterwarnings('ignore')
import numpy as np


# %%
logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)
url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'


# %%
data = pd.read_csv(url,delimiter=';')


# %%
X = data.iloc[:,:11]
y = data.iloc[:,-1]
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=10)
n_jobs=5
normalize=True
#log params
#mlflow.log_param('n_jobs',n_jobs)
#mlflow.log_param('normalize',normalize)


# %%
model = LinearRegression(n_jobs=n_jobs,normalize=normalize)
model.fit(X_train,y_train)
y_pred = model.predict(X_test)


# %%
mse = mean_squared_error(y_test,y_pred)
r2 = r2_score(y_test,y_pred)
n = X_test.shape[0]
p = X_test.shape[1]
adjr2 = 1-(1-r2) * (n-1)/(n-p-1)
#mlflow.log_metric('mse',mse)
#mlflow.log_metric('r2',r2)
#mlflow.log_metric('adjr2',adjr2)


# %%
#mlflow.sklearn.log_model(model,artifact_path='sklearn-model')


# %%
print('r2 : {0}, mse : {1}, adjr2 : {2}',r2,mse,adjr2)


