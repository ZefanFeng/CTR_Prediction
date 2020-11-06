import xlearn as xl

ffm_model = xl.create_ffm()
ffm_model.setTrain('./ffm_dataset2.txt')
ffm_model.setValidate("./ffm_dataset_test2.txt")
param = {'task':'binary', 'lr':0.2,
         'lambda':0.0005, 'metric':'f1',
         'opt':'sgd','epoch':15}

ffm_model.fit(param, "ffm_model.out")