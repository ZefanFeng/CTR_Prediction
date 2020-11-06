import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
import warnings

warnings.filterwarnings('ignore')

new_train = pd.read_csv('/tmp/pycharm_project_156/new_train.csv',index_col=0)
new_test = pd.read_csv('/tmp/pycharm_project_156/new_test.csv',index_col=0)


def convert_obj_to_int(self):
    object_list_columns = self.columns
    object_list_dtypes = self.dtypes
    new_col_suffix = '_int'
    for index in range(0, len(object_list_columns)):
        if object_list_dtypes[index] == object:
            self[object_list_columns[index] + new_col_suffix] = self[object_list_columns[index]].map(lambda x: hash(x))
            self.drop([object_list_columns[index]], inplace=True, axis=1)
    return self

train_num = convert_obj_to_int(new_train)

# divide dataset into the label and predictors
click_train = train_num['click']
del train_num['click']

test_num = convert_obj_to_int(new_test)
click_test = test_num['click']
del test_num['click']

poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
interact_train = pd.DataFrame(poly.fit_transform(train_num))
interact_test = pd.DataFrame(poly.fit_transform(test_num))

train_inter = pd.merge(train_num, interact_train, left_index = True, right_index = True, how = 'inner')
test_inter = pd.merge(test_num, interact_test, left_index = True, right_index = True, how = 'inner')

train_standard = train_inter.copy(deep = True)

#implement normalization
train_standard = train_standard.apply(lambda x : (x-np.mean(x))/np.std(x))

#add label the normalizaed dataset
train_standard['click'] = click_train
train_standard.head()

test_standard = test_inter.copy(deep = True)
test_standard = test_standard.apply(lambda x : (x-np.mean(x))/np.std(x))
test_standard['click'] = click_test
test_standard.head()

df_x = train_standard.iloc[:,0:252]
df_y = train_standard.iloc[:,252:253]

test_x = train_standard.iloc[:,0:252]
test_y = train_standard.iloc[:,252:253]

#  reduce the dimention from 252 into 22 so we will get 22 components
estimator = PCA(n_components=22)

X_pca=estimator.fit_transform(df_x)
test_X_pca = estimator.fit_transform(test_x)

# C is the coeffcient of the penalty item.
lr = LogisticRegression(fit_intercept = True, C = 1).fit(X_pca,df_y)

# calculate the accuracy of model
print(lr.score(test_X_pca,test_y))





