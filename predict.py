from Dataset import Dataset
from Discova import Discova
import numpy as np
import pandas as pd

results = pd.DataFrame()
# train house on rows, test house on cols

fridgeLabels = ['05 refrigerator',
                '09 refrigerator',
                '07 refrigerator',
                '14 kitchen_outlets',
                '18 refrigerator',
                '08 refrigerator']

for train_house in range(1,7):
    discova = Discova(eps_std = 0) # Turn off variance for prediction
    discova.construct_model()
    discova.compile_model()
    path = 'weights/weights_0' + str(train_house) + '_fridge.hdf5'
    discova.model.load_weights(path)

    for test_house in range(1,7):
        load_str = 'Loading test data for house ' + str(test_house) + '...'
        print(load_str.ljust(40,'.'))
        fridgeLabel = fridgeLabels[test_house - 1]
        data = Dataset()
        data.load_house_dataframe(test_house)
        data.add_windows(test_house, '00 mains')
        data.add_windows(test_house, fridgeLabel)
        data.add_statistics(test_house)
        print('Predicting'.ljust(40,'.'))
        predRaw = discova.model.predict(data.windows[test_house]['00 mains'])
        print('Aligning by timestep'.ljust(40,'.'))
        predTimestep = data.recover_reverse_diagonals(predRaw)
        print('Taking median'.ljust(40,'.'))
        predMedian = np.median(predTimestep, axis = 1)
        print('Rescaling'.ljust(40,'.'))
        mean = data.means[test_house][fridgeLabel]
        std = data.stddevs[test_house][fridgeLabel]
        predMedianScaled = predMedian * std + mean
        print('Calculating relative MAE'.ljust(40,'.'))
        true = data.dataframes[test_house][fridgeLabel][511:-511].as_matrix()
        mae = np.abs(true - predMedianScaled)
        relative_mae = mae.sum() / true.sum()
        results.loc[train_house, test_house] = relative_mae