import traceback
import numpy as np
import pandas as pd

from keras.models import Model
from sklearn.metrics.pairwise import paired_cosine_distances

from Dataset import Dataset
from Discova import Discova
from Autoencoder import Autoencoder
from Prediction import Prediction


class LatentSpace(Prediction):

    def evaluate_latent_space(self, networkType):

        for appliance in self.applianceTypes:
            print('Adding results for {}'.format(appliance).ljust(40,'.'))
            for house in self.houses:
                try:
                    network = self.getNetworkWithWeights(networkType, appliance, house)
                    latent_model = Model(
                        inputs = network.model.input,
                        outputs = network.model.layers[13].output
                        )

                    load_str = 'Loading test data for House ' + str(house)
                    print(load_str.ljust(40,'.'))
                    data = Dataset()
                    data.load_house_dataframe(house)
                    data.add_windows(house, '00 mains')
                    mainsWindows = self.getMainsWindows(data, house, outOfHouse = False)

                    print('Getting latent vectors'.ljust(40,'.'))
                    latent_vecs = latent_model.predict(mainsWindows)

                    print('Calculating neighbor distance'.ljust(40,'.'))
                    neighbor_diff = latent_vecs[1:] - latent_vecs[:-1]
                    mean_latent_dist = np.linalg.norm(neighbor_diff, axis = 1).mean()

                    print('Calculating neighbor cosine similarity'.ljust(40,'.'))
                    mean_cosine_sim = paired_cosine_distances(
                        latent_vecs[1:],
                        latent_vecs[:-1]
                        ).mean()

                    print('Appending results'.ljust(40,'.'))
                    row_dict = {'network': networkType,
                                'appliance': appliance,
                                'train_house': house,
                                'predict_house': house,
                                'euclidean_dist': mean_latent_dist,
                                'cosine_similarity': mean_cosine_sim}
                except Exception:
                    traceback.print_exc()
                    row_dict = {'network': networkType,
                                'appliance': appliance,
                                'train_house': house,
                                'predict_house': house,
                                'euclidean_dist': None,
                                'cosine_similarity': None}
                    print('Error on House {} {}'.format(house, appliance).ljust(40,'.'))

                self.results = self.results.append(row_dict, ignore_index=True)


#latent = LatentSpace()
#latent.evaluate_latent_space('vae')