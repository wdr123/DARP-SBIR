from dataset_chairv2 import *
import time
import itertools
import torch.nn.functional as F
import torch.nn as nn
from matplotlib import pyplot as plt
import numpy as np
from torch.distributions.multivariate_normal import MultivariateNormal
import math
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pi = torch.FloatTensor([math.pi]).to(device)
rgb_dir = './data/chairv2'



class Environment():
    def __init__(self):

        parser = argparse.ArgumentParser()
        opt = parser.parse_args()
        opt.coordinate = 'ChairV2_Coordinate'
        opt.roor_dir = rgb_dir
        opt.mode = 'Train'
        opt.Train = True
        opt.shuffle = False
        opt.nThreads = 2
        opt.batch_size = 1
        #
        dataset_sketchy_train = CreateDataset_Sketchy(opt, on_Fly=True)
        dataloader_sketchy_train = data.DataLoader(dataset_sketchy_train, batch_size=opt.batch_size, shuffle=opt.shuffle,
                                                   num_workers=int(opt.nThreads))


        # self.Sketch_Array_Train = []

        self.Sketch_Name_Train = []

        for i_batch, sanpled_batch in enumerate(dataloader_sketchy_train):
            # sketch_feature_ALL = torch.FloatTensor().to(device)
            # for data_sketch in sanpled_batch['sketch_img']:
            #     sketch_feature = data_sketch.to(device) # 1*3*299*299
            #     sketch_feature_ALL = torch.cat((sketch_feature_ALL, sketch_feature.detach()))
            self.Sketch_Name_Train.extend(sanpled_batch['sketch_path'])
            # self.Sketch_Array_Train.append(sketch_feature_ALL.cpu())
            print("Sketch training complete{}".format(i_batch))

        parser = argparse.ArgumentParser()
        test_opt = parser.parse_args()
        test_opt.coordinate = 'ChairV2_Coordinate'
        test_opt.roor_dir = rgb_dir
        test_opt.mode = 'Test'
        test_opt.Train = False
        test_opt.shuffle = False
        test_opt.nThreads = 2
        test_opt.batch_size = 1

        dataset_sketchy_test = CreateDataset_Sketchy(test_opt, on_Fly=True)
        dataloader_sketchy_test = data.DataLoader(dataset_sketchy_test, batch_size=test_opt.batch_size,
                                                  shuffle=test_opt.shuffle,
                                                  num_workers=int(test_opt.nThreads))


        # self.Sketch_Array_Test = []

        self.Sketch_Name_Test = []

        for i_batch, sanpled_batch in enumerate(dataloader_sketchy_test):

            # sketch_feature_ALL = torch.FloatTensor().to(device)
            # for data_sketch in sanpled_batch['sketch_img']:
            #     sketch_feature = data_sketch.to(device)
            #     sketch_feature_ALL = torch.cat((sketch_feature_ALL, sketch_feature.detach()))
            self.Sketch_Name_Test.extend(sanpled_batch['sketch_path'])
            # self.Sketch_Array_Test.append(sketch_feature_ALL.cpu())
            print("Sketch testing complete{}".format(i_batch))

        with open("sketch_train.pickle", "wb") as f:
            pickle.dump(self.Sketch_Name_Train, f)

        with open("sketch_test.pickle", "wb") as f:
            pickle.dump(self.Sketch_Name_Test, f)


if __name__ == "__main__":
    env = Environment()