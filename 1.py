import xlearn as xl


lr_model = xl.create_linear()
lr_model.setTrain('./ffm_dataset2.txt')

# using validation to do the prediction on tets data and assess the model's performance
lr_model.setValidate("./ffm_dataset_test2.txt")

param = {'task':'binary', 'lr':0.2,
         'lambda':0.2, 'metric':'f1',
         'opt':'sgd'}

lr_model.fit(param, "lr_model.out")