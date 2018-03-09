### Import packages ###

# Standard packages
import numpy as np
import pandas as pd

# Sklearn utils
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler

class Dataset:

    def __init__(self):
        self.dataframes = {}
        self.windows = {}
        self.scalers = {}
        self.skip_threshold = 30
        self.pad_size = 0
        self.window_size = 512

    def load_house_dataframe(self, house):
        df = pd.read_csv('clean_data/house_' + str(house) + '.csv')
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index(['Date'])
        self.dataframes[house] = df

    def add_timestep_col(self, house, return_array = False):
        times = self.dataframes[house].index.to_series()
        timesteps = times.diff().dt.total_seconds()
        self.dataframes[house]['timestep'] = timesteps
        if return_array:
            return timesteps.as_matrix()

    def add_mains_col(self, house, return_array = False):
        columns = self.dataframes[house].columns
        mains = [col for col in columns if 'mains' in col]
        sum_mains = self.dataframes[house][mains].sum(axis = 1)
        self.dataframes[house]['00 mains'] = sum_mains
        if return_array:
            return sum_mains.as_matrix()

    def add_scaler_object(self, house):
        scaler = StandardScaler()
        df = self.dataframes[house]
        scaler.fit(df)
        self.scalers[house] = scaler

    def recover_reverse_diagonals(self, array):
        m, n = array.shape
        rows = np.arange(n)[None, :] + np.arange(m-n+1)[:, None]
        cols = np.arange(n-1,-1,-1)
        return array[(rows, cols)]

    def add_windows(self, house, appliance):

        # Skip mask
        timesteps = self.dataframes[house]['timestep'].as_matrix()
        skip_array = self.create_skip_mask(timesteps, self.skip_threshold)

        # Appliance array
        appliance_series = self.dataframes[house][appliance]
        appliance_array = appliance_series.as_matrix().reshape(-1, 1)
        appliance_array = self.get_scaled_array(appliance_array)

        # Windowizing
        print('Removing timesteps above threshold...')
        appliance_array = self.split_by_skips(appliance_array, skip_array)
        print('Dropping arrays smaller than window...')
        window_size = 512 - 2 * self.pad_size
        array_list = self.drop_small_arrays(appliance_array)
        print('Creating windowed data...')
        appliance_array = np.squeeze(self.create_input_array(array_list))

        if self.pad_size > 0:
            print('Padding with zeros...')
            appliance_array = self.left_right_pad(appliance_array, 0, pad_size)

        if house not in self.windows.keys():
            self.windows[house] = {}
        
        self.windows[house][appliance] = appliance_array

    def format_for_keras(
        self,
        house,                  # House number as int
        appliance,              # Appliance as string e.g. '05 refrigerator'
        sample_size = 0.1,      # Proportion of total data to keep
        test_size = 0.2,        # Portion of sampled data to keep
        random_state = 123,     # Random seed for reproduction
        vae = True):            # Flag for VAE
        
        mains_array = self.windows[house]['00 mains']
        appliance_array = self.windows[house][appliance]

        print('Sampling data...')
        n_samples = int(mains_array.shape[0]*sample_size)
        X, Y = resample(mains_array,
                        appliance_array,
                        n_samples = n_samples,
                        random_state = random_state)
        
        # 3.2 Create train and validation splits
        print('Creating train / test splits...')
        X_train, X_val, Y_train, Y_val = \
            train_test_split(   X,
                                Y,
                                test_size = test_size,
                                random_state = random_state)

        # 3.3 Reformat for keras
        if vae == False:
            print('Reshaping...')
            X_train = X_train[:, :, np.newaxis]
            X_val = X_val[:, :, np.newaxis]
            Y_train = Y_train[:, :, np.newaxis]
            Y_val = Y_val[:, :, np.newaxis]

        print('Finished.')
        return X_train, X_val, Y_train, Y_val

    def get_scaled_array(self, dataframe, scaler = StandardScaler()):
        scaler.fit(dataframe)
        scaled_array = scaler.transform(dataframe)
        return scaled_array

    def create_skip_mask(self, timestep_array, skip_threshold):
        skip_mask = np.nan_to_num(timestep_array) > skip_threshold
        return skip_mask

    def split_by_skips(self, array, skip_mask):
        '''
        Takes a timeseries array and splits an array
        into a list of subarrays separated by timestep skips
        '''
        # Find indices where there there is a change from 
        # normal timesteps to skipped timestep or vice versa
        indices = np.nonzero(skip_mask[1:] != skip_mask[:-1])[0] + 1
        # Split array into sub arrays at these indices (subarrays 
        # will be normal, skipped, normal, skipped etc.)
        split_array = np.split(array, indices)
        # Only keep arrays where the data does not contain any time series skips
        if not skip_mask[0]:
            split_array = split_array[0::2]
        else:
            split_array = split_array[1::2]

        return split_array

    def drop_small_arrays(self, array_list):
        #small_arrays = [subarray in array_list if len(subarray) < window_size]
        #data_points_dropped = sum([len(subarray) for subarray in small_arrays])
        #print('{} data points will be dropped.'.format(data_points_dropped))
        big_arrays = [a for a in array_list if len(a) > self.window_size]
        return big_arrays

    def create_sliding_windows(self, array):
        timesteps = len(array)
        rows = np.arange(timesteps - self.window_size + 1)[:, None]
        cols = np.arange(self.window_size)[None,:]
        indices = rows + cols
        return array[indices]

    def create_input_array(self, array_list):
        array_windows_list = []
        for subarray in array_list:
            subarray_windows = self.create_sliding_windows(subarray)
            array_windows_list.append(subarray_windows)
        input_array = np.concatenate(array_windows_list, axis = 0)
        return input_array

    def left_right_pad(self, array, pad_num, pad_size):
        pad = np.full((array.shape[0], pad_size), pad_num)
        return np.concatenate([pad, array, pad], axis = 1)
