from Dataset import Dataset
from Discova import Discova
from Autoencoder import Autoencoder

import numpy as np
import pandas as pd

results = pd.DataFrame()
# train house on rows, test house on cols

dfLabels = pd.read_csv('appliance_codes.csv')

applianceLabels = dfLabels['Washer Dryer'].tolist()

for train_house in [5]:
    discova = Autoencoder() # Turn off variance for prediction
    discova.construct_model()
    discova.compile_model()
    path = 'weights/ae_0' + str(train_house) + '_washerdryer.hdf5'
    discova.model.load_weights(path)
    print(('Loading weights for House ' + str(train_house)).ljust(40,'.'))

    for test_house in range(1,7):
        load_str = 'Loading test data for House ' + str(test_house)
        print(load_str.ljust(40,'.'))
        applianceLabel = applianceLabels[test_house - 1]
        data = Dataset()
        data.load_house_dataframe(test_house)
        data.add_windows(test_house, '00 mains')
        data.add_windows(test_house, applianceLabel)
        data.add_statistics(test_house)
        print('Predicting'.ljust(40,'.'))
        predRaw = discova.model.predict(data.windows[test_house]['00 mains'])
        print('Aligning by timestep'.ljust(40,'.'))
        predTimestep = data.recover_reverse_diagonals(predRaw)
        print('Taking median'.ljust(40,'.'))
        predMedian = np.median(predTimestep, axis = 1)
        print('Rescaling'.ljust(40,'.'))
        mean = data.means[test_house][applianceLabel]
        std = data.stddevs[test_house][applianceLabel]
        predMedianScaled = predMedian * std + mean
        print('Calculating relative MAE'.ljust(40,'.'))
        true = data.dataframes[test_house][applianceLabel][511:-511].as_matrix()
        mae = np.abs(true - predMedianScaled)
        relative_mae = mae.sum() / true.sum()
        results.loc[train_house, test_house] = relative_mae