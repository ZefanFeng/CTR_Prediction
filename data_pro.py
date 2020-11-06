import numpy as np
import pandas as pd
import random
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import xlearn as xl

train = pd.read_csv('./train.csv')
test = pd.read_csv('./test.csv')
print(train.head().shape)
print(train['click'].value_counts())
print(test['click'].value_counts())

def downsampling(dataset, num_0, num_1):
    ran = range(0, num_0)
    # implement randomly choose
    nums = random.sample(ran, num_1)
    sub_data = pd.DataFrame()

    # add all selected records into sub_dataset
    for i in range(0, num_1):
        n = dataset.iloc[nums[i], :]
        sub_data = sub_data.append(n)
    return sub_data
sub_train = train[train['click']==0]
# select out 20792 records from 98214 train records whose 'click' equals to 0
sub_train = downsampling(sub_train, 98214, 20792)
print(sub_train.shape)


sub_test = test[test['click']==0]

# select out 23873 records from 113569 test records whose 'click' equals to 0
sub_test = downsampling(sub_test, 113569, 23873)

print(sub_test.shape)

new_train = pd.DataFrame()
new_test = pd.DataFrame()

new_train = pd.concat([sub_train, train[train['click']==1]], axis = 0)
new_test = pd.concat([sub_test, test[test['click']==1]], axis = 0)

#reset the index of concatenated dataset
new_train = new_train.reset_index(drop=True)
new_test = new_test.reset_index(drop=True)

new_train.to_csv('./new_train.csv')
new_test.to_csv('./new_test.csv')

