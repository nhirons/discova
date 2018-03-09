from Dataset import Dataset
from Discova import Discova

data = Dataset()
data.load_house_dataframe(1)
data.add_windows(1, '00 mains')
data.add_windows(1, '05 refrigerator')
X_train, X_val, Y_train, Y_val = data.format_for_keras(1, '05 refrigerator')

'''
discova = Discova()
discova.load_data(X_train, X_val, Y_train, Y_val)
discova.construct_model()
discova.compile_model()
discova.load_tensorboard()
discova.train_model()
'''
