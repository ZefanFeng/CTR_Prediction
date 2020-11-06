import xlearn as xl
fm_model = xl.create_fm()
fm_model.setTrain('./ffm_dataset2.txt')
fm_model.setValidate("./ffm_dataset_test2.txt")
param = {'task':'binary', 'lr':0.2,
         'lambda':0.001, 'metric':'f1',
         'opt':'sgd','epoch':15}
fm_model.fit(param, "fm_model.out")