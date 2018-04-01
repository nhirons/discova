import traceback

import numpy as np
import pandas as pd

from Dataset import Dataset
from Discova import Discova
from Autoencoder import Autoencoder


class Prediction:

    def __init__(self):
        self.dfLabels = pd.read_csv('appliance_codes.csv', index_col = 0)
        self.applianceTypes = self.dfLabels.columns.tolist()
        self.houses = list(range(1,7))
        self.results = pd.DataFrame()

    def getNetworkWithWeights(self, networkType, appliance, house):
        print(('Loading weights for House ' + str(house)).ljust(40,'.'))
        if networkType == 'vae':
            network = Discova()
            weightPath = 'weights/weights_0{}_{}.hdf5'.format(house, appliance)
        else:
            network = Autoencoder()
            weightPath = 'weights/ae_0{}_{}.hdf5'.format(house, appliance)
        print('Weights loaded. Compiling model.'.ljust(40,'.'))
        network.construct_model()
        network.compile_model()
        network.model.load_weights(weightPath)
        return network

    def getMainsWindows(self, data, house, outOfHouse = True):
        if outOfHouse == True:
            mainsWindows = data.windows[house]['00 mains']
        else:
            m = data.dataframes[house].shape[0]
            startIdx = int(-0.2 * m)
            mainsWindows = data.windows[house]['00 mains'][startIdx:]
        return mainsWindows

    def evaluateInHouse(self, networkType):

        for appliance in self.applianceTypes:
            print('Adding results for {}'.format(appliance).ljust(40,'.'))
            for house in self.houses:
                try:
                    network = self.getNetworkWithWeights(networkType, appliance, house)

                    load_str = 'Loading test data for House ' + str(house)
                    print(load_str.ljust(40,'.'))
                    data = Dataset()
                    data.load_house_dataframe(house)
                    data.add_windows(house, '00 mains')
                    mainsWindows = self.getMainsWindows(data, house, outOfHouse = False)

                    print('Predicting'.ljust(40,'.'))
                    predRaw = network.model.predict(mainsWindows)

                    print('Aligning by timestep'.ljust(40,'.'))
                    predTimestep = data.recover_reverse_diagonals(predRaw)

                    print('Taking median'.ljust(40,'.'))
                    predMedian = np.median(predTimestep, axis = 1)

                    print('Rescaling'.ljust(40,'.'))
                    data.add_statistics(house)
                    applianceLabel = self.dfLabels.loc[house, appliance]
                    mean = data.means[house][applianceLabel]
                    std = data.stddevs[house][applianceLabel]
                    predMedianScaled = predMedian * std + mean

                    print('Calculating relative MAE'.ljust(40,'.'))
                    data.add_windows(house, applianceLabel)
                    m  = data.dataframes[house].shape[0]
                    testStartIdx = int(-0.2 * m)
                    true = data.dataframes[house][applianceLabel]
                    true = true[testStartIdx:-511].as_matrix()
                    mae = np.abs(true - predMedianScaled)
                    relative_mae = mae.sum() / true.sum()

                    print('Appending results'.ljust(40,'.'))
                    row_dict = {'network': networkType,
                                'appliance': appliance,
                                'train_house': house,
                                'predict_house': house,
                                'relative_mae': relative_mae}
                except Exception:
                    traceback.print_exc()
                    row_dict = {'network': networkType,
                                'appliance': appliance,
                                'train_house': house,
                                'predict_house': house,
                                'relative_mae': None}
                    print('Error on House {} {}'.format(house, appliance).ljust(40,'.'))

                self.results = self.results.append(row_dict, ignore_index=True)


predictor = Prediction()
predictor.evaluateInHouse('ae')