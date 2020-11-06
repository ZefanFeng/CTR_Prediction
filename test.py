import xlearn as xl
ffm_model = xl.create_ffm() 
ffm_model.setSign()
ffm_model.setTest("./ffm_dataset_test2.txt")
ffm_model.predict("./ffm_model.out", "./output.txt")
