import torch

import utils
import argparse

from SBIR_trainer import Trainer
from config import get_config
from dataset_chairv2 import *


def main(config):
    utils.prepare_dirs(config)

    # ensure reproducibility
    torch.manual_seed(config.random_seed)
    kwargs = {}
    if config.use_gpu:
        torch.cuda.manual_seed(config.random_seed)
        kwargs = {"num_workers": 1, "pin_memory": True}

    # instantiate data loaders
    parser = argparse.ArgumentParser()
    opt = parser.parse_args()
    opt.coordinate = 'ChairV2_Coordinate'
    opt.roor_dir = './data/chairv2'
    opt.mode = 'Train'
    opt.Train = True
    opt.shuffle = False
    opt.nThreads = 4
    opt.batch_size = 32
    dataset_sketchy_train = CreateDataset_Sketchy(opt, on_Fly=True)
    dataloader_sketchy_train = data.DataLoader(dataset_sketchy_train, batch_size=opt.batch_size,
                                               shuffle=opt.shuffle,
                                               num_workers=int(opt.nThreads))

    parser = argparse.ArgumentParser()
    test_opt = parser.parse_args()
    test_opt.coordinate = 'ChairV2_Coordinate'
    test_opt.roor_dir = './data/chairv2'
    test_opt.mode = 'Test'
    test_opt.Train = False
    test_opt.shuffle = False
    test_opt.nThreads = 4
    test_opt.batch_size = 1

    dataset_sketchy_test = CreateDataset_Sketchy(test_opt, on_Fly=True)
    dataloader_sketchy_test = data.DataLoader(dataset_sketchy_test, batch_size=test_opt.batch_size,
                                              shuffle=test_opt.shuffle,
                                              num_workers=int(test_opt.nThreads))

    # parser = argparse.ArgumentParser()
    # valid_opt = parser.parse_args()
    # valid_opt.coordinate = 'ChairV2_Coordinate'
    # valid_opt.roor_dir = './data/chairv2'
    # valid_opt.mode = 'Valid'
    # valid_opt.Train = False
    # valid_opt.shuffle = False
    # valid_opt.nThreads = 4
    # valid_opt.batch_size = 1
    # dataset_sketchy_valid = CreateDataset_Sketchy(valid_opt, on_Fly=True)
    # # print(len(dataset_sketchy_valid))
    # dataloader_sketchy_valid = data.DataLoader(dataset_sketchy_valid, batch_size=valid_opt.batch_size,
    #                                           shuffle=valid_opt.shuffle,
    #                                           num_workers=int(valid_opt.nThreads))

    if config.is_train:
        # dloader = data_loader.get_train_valid_loader(
        #     config.data_dir,
        #     config.batch_size,
        #     config.random_seed,
        #     config.valid_size,
        #     config.shuffle,
        #     config.show_sample,
        #     **kwargs,
        # )
        dloader = (dataloader_sketchy_train, dataloader_sketchy_test)

    else:
        # dloader = data_loader.get_test_loader(
        #     config.data_dir, config.batch_size, **kwargs,
        # )
        dloader = dataloader_sketchy_test

    # data_iter = iter(dloader[0])
    # data_iter.__next__()
    # images = data_iter.next()
    # X = images['sketch_img'].numpy()
    # np.set_printoptions(threshold=np.inf)
    # print('X', X[0])
    # print('max', X[0].max())
    # print('min', X[0].min())
    trainer = Trainer(config, dloader)

    # either train
    if config.is_train:
        utils.save_config(config)
        trainer.train()
    # or load a pretrained model and test
    else:
        trainer.test1()


if __name__ == "__main__":
    config, unparsed = get_config()
    main(config)
