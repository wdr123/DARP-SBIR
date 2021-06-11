import random
import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import torchvision.transforms.functional as F
import argparse
import pickle
import os
import time
import numpy as np
from random import randint
from PIL import Image
import torchvision
from render_sketch_chairv2_64 import redraw_Quick2RGB
from utils import plot_images_back

class AddGaussianNoise(object):

    def __init__(self, mean=0.0, variance=1.0, amplitude=1.0):

        self.mean = mean
        self.variance = variance
        self.amplitude = amplitude

    def __call__(self, img):
        img = np.array(img)
        if(len(img.shape)==2):
            h, w = img.shape
            N = self.amplitude * np.random.normal(loc=self.mean, scale=self.variance, size=(h, w))
        else:
            h, w, c= img.shape
            N = self.amplitude * np.random.normal(loc=self.mean, scale=self.variance, size=(h, w, 1))
            N = np.repeat(N, c, axis=2)

        img = N + img
        img[img > 255] = 255                    # 避免有值超过1而反转
        img[img < 0] = 0

        img = Image.fromarray(img.astype('uint8'))
        return img

def get_ransform(opt):
    transform_list = []
    if opt.Train:
        # n_rotate = random.random()
        # if n_rotate > 0.5:
        transform_list.extend([
        transforms.RandomRotation((-10,10), resample=2)])
            # n_crop = random.random()
            # if n_crop > 0.5:
            #     transform_list.extend([transforms.Resize(32), transforms.CenterCrop(28)])
            # elif n_crop > 0.3:
            #     transform_list.extend([transforms.Resize(64), transforms.CenterCrop(28)])
            # else:
            #     transform_list.extend([transforms.RandomResizedCrop(size=28, scale=(0.7, 1.0))])
        # else:
        transform_list.extend([transforms.Resize(32), transforms.CenterCrop(28)])
    else:
        transform_list.extend([transforms.Resize(28)])
    transform_list.extend(
        # [transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    return transforms.Compose(transform_list)


class CreateDataset_Sketchy(data.Dataset):
    def __init__(self, opt, on_Fly=False, show_sample=False, print2loc=True):
        with open(os.path.join(opt.roor_dir,opt.coordinate), 'rb') as fp:
            self.Coordinate = pickle.load(fp)

        self.Skecth_Train_List = [x for x in self.Coordinate if 'train' in x]
        self.Skecth_Test_List = [x for x in self.Coordinate if 'test' in x]
        self.Skecth_Valid_List = self.Skecth_Test_List[:100]
        self.Skecth_Test_List = self.Skecth_Test_List[100:]


        self.opt = opt
        self.transform = get_ransform(opt)
        self.on_Fly = on_Fly
        self.show_sample = show_sample
        self.printout = print2loc


    def __getitem__(self, item):

        if self.opt.mode == 'Train':
            sketch_path = self.Skecth_Train_List[item]

            positive_sample =  '_'.join(self.Skecth_Train_List[item].split('/')[-1].split('_')[:-1])
            positive_path = os.path.join(self.opt.roor_dir, 'photo', positive_sample + '.png')
            possible_list = list(range(len(self.Skecth_Train_List)))
            possible_list.remove(item)
            negetive_item = possible_list[randint(0, len(possible_list) - 1)]
            negetive_sample = '_'.join(self.Skecth_Train_List[negetive_item].split('/')[-1].split('_')[:-1])
            negetive_path = os.path.join(self.opt.roor_dir, 'photo', negetive_sample + '.png')

            vector_x = self.Coordinate[sketch_path]
            sketch_img, Sample_len = redraw_Quick2RGB(vector_x)

            if self.on_Fly == False:
                sketch_img = Image.fromarray(sketch_img[-1].astype('uint8'))
            else:
                sketch_img = [Image.fromarray(sk_img.astype('uint8')) for sk_img in sketch_img]

            positive_img = Image.open(positive_path)
            negetive_img = Image.open(negetive_path)

            if self.printout:
                if not self.on_Fly:
                    print('yes')
                    positive_img.save('./posiImg.png')
                    negetive_img.save('./negImg.png')
                    sketch_img.save('./sketch.png')
                else:
                    print('multi')
                    parent_dir = './data_sample/Group{}'.format(item)
                    if not os.path.exists(parent_dir):
                        os.mkdir(parent_dir)
                    for i in range(17):
                        sketch_img[i].save(parent_dir+'/sketch{}.png'.format(i))
                    positive_img.save(parent_dir+'/posiImg.png')
                    negetive_img.save(parent_dir+'/negImg.png')

            n_flip = random.random()
            if n_flip > 0.5:

                if self.on_Fly == False:
                    sketch_img = F.hflip(sketch_img)
                else:
                    sketch_img = [F.hflip(sk_img) for sk_img in sketch_img]

                positive_img = F.hflip(positive_img)
                negetive_img = F.hflip(negetive_img)

            # elif n_flip > 0.4:
            #     if self.on_Fly == False:
            #         sketch_img = F.vflip(sketch_img)
            #     else:
            #         sketch_img = [F.vflip(sk_img) for sk_img in sketch_img]
            #
            #     positive_img = F.vflip(positive_img)
            #     negetive_img = F.vflip(negetive_img)

            if self.on_Fly == False:
                sketch_img = self.transform(sketch_img)
            else:
                sketch_img = [self.transform(sk_img) for sk_img in sketch_img]
                # sketch_img_back = []
                # for index, sk_img in enumerate(sketch_img):
                #     sk_img = self.transform(sk_img)
                #     # sk_img = F.normalize(sk_img, mean=[torch.mean(sk_img).item(),], std=[torch.std(sk_img).item(),])
                #     sketch_img_back.append(sk_img)
                #
                # sketch_img = sketch_img_back


            if(self.show_sample):
                if self.on_Fly:
                    sampled = sketch_img[0:9]
                    X = torch.stack(sampled)
                else:
                    X = sketch_img.unsqueeze(0)

                # print('x：',X.numpy().max())
                X = np.transpose(X, [0, 2, 3, 1])
                plot_images_back(X, self.on_Fly)


            positive_img = self.transform(positive_img)
            negetive_img = self.transform(negetive_img)


            sample = {'sketch_img': sketch_img, 'sketch_path': self.Skecth_Train_List[item],
                      'positive_img': positive_img, 'positive_path': positive_sample,
                      'negetive_img': negetive_img, 'negetive_path': negetive_sample,
                      'Sample_Len': Sample_len}


        elif self.opt.mode == 'Test':
            sketch_path = self.Skecth_Test_List[item]

            positive_sample = '_'.join(self.Skecth_Test_List[item].split('/')[-1].split('_')[:-1])
            positive_path = os.path.join(self.opt.roor_dir, 'photo', positive_sample + '.png')
            possible_list = list(range(len(self.Skecth_Test_List)))
            possible_list.remove(item)
            negetive_item = possible_list[randint(0, len(possible_list) - 1)]
            negetive_sample = '_'.join(self.Skecth_Test_List[negetive_item].split('/')[-1].split('_')[:-1])
            negetive_path = os.path.join(self.opt.roor_dir, 'photo', negetive_sample + '.png')
            vector_x = self.Coordinate[sketch_path]
            sketch_img, Sample_len = redraw_Quick2RGB(vector_x)

            if self.on_Fly == False:
                sketch_img = self.transform(Image.fromarray(sketch_img[-1].astype('uint8')))
            else:
                sketch_img = [self.transform(Image.fromarray(sk_img.astype('uint8'))) for sk_img in sketch_img]
            # if self.on_Fly == False:
            #     sketch_img = self.transform(sketch_img)
            #     # sketch_img = F.normalize(sketch_img, mean=torch.mean(sketch_img).item(), std=torch.std(sketch_img).item())
            # else:
            #     sketch_img_back = []
            #     for index, sk_img in enumerate(sketch_img):
            #         sk_img = self.transform(sk_img.astype('uint8'))
            #         # sk_img = F.normalize(sk_img, mean=torch.mean(sk_img).item(), std=torch.std(sk_img).item())
            #         sketch_img_back.append(sk_img)
            #     sketch_img = sketch_img_back

            positive_img = self.transform(Image.open(positive_path))
            negetive_img = self.transform(Image.open(negetive_path))

            sample = {'sketch_img': sketch_img, 'sketch_path': self.Skecth_Test_List[item],
                      'positive_img': positive_img,
                      'negetive_img': negetive_img, 'negetive_path': negetive_sample,
                      'positive_path': positive_sample, 'Sample_Len': Sample_len}

        elif self.opt.mode == 'Valid':
            sketch_path = self.Skecth_Valid_List[item]

            positive_sample = '_'.join(self.Skecth_Valid_List[item].split('/')[-1].split('_')[:-1])
            positive_path = os.path.join(self.opt.roor_dir, 'photo', positive_sample + '.png')
            possible_list = list(range(len(self.Skecth_Valid_List)))
            possible_list.remove(item)
            negetive_item = possible_list[randint(0, len(possible_list) - 1)]
            negetive_sample = '_'.join(self.Skecth_Valid_List[negetive_item].split('/')[-1].split('_')[:-1])
            negetive_path = os.path.join(self.opt.roor_dir, 'photo', negetive_sample + '.png')
            vector_x = self.Coordinate[sketch_path]
            sketch_img, Sample_len = redraw_Quick2RGB(vector_x)

            if self.on_Fly == False:
                sketch_img = self.transform(Image.fromarray(sketch_img[-1].astype('uint8')))
            else:
                sketch_img = [self.transform(Image.fromarray(sk_img.astype('uint8'))) for sk_img in sketch_img]
            # if self.on_Fly == False:
            #     sketch_img = self.transform(sketch_img.astype('uint8'))
            # else:
            #     sketch_img_back = []
            #     for index, sk_img in enumerate(sketch_img):
            #         sk_img = self.transform(sk_img.astype('uint8'))
            #         # sk_img = F.normalize(sk_img, mean=torch.mean(sk_img).item(), std=torch.std(sk_img).item())
            #         sketch_img_back.append(sk_img)
            #         # print('mean{}:{}'.format(index, torch.mean(sk_img).item()))
            #         # print('std{}:{}'.format(index, torch.std(sk_img).item()))
            #         # print('img_trans{}:'.format(index), sk_img)
            #     sketch_img = sketch_img_back


            positive_img = self.transform(Image.open(positive_path))
            negetive_img = self.transform(Image.open(negetive_path))

            sample = {'sketch_img': sketch_img, 'sketch_path': self.Skecth_Valid_List[item],
                      'positive_img': positive_img,
                      'negetive_img': negetive_img, 'negetive_path': negetive_sample,
                      'positive_path': positive_sample, 'Sample_Len': Sample_len}

        return sample

    def __len__(self):
        if self.opt.mode == 'Train':
            return len(self.Skecth_Train_List)
        elif self.opt.mode == 'Test':
            return len(self.Skecth_Test_List)
        elif self.opt.mode == 'Valid':
            return len(self.Skecth_Valid_List)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    opt = parser.parse_args()
    opt.coordinate = 'ChairV2_Coordinate'
    opt.roor_dir = './'
    opt.mode = 'Train'
    opt.Train = True
    opt.shuffle = True
    opt.nThreads = 2
    opt.batchsize = 16
    dataset_sketchy = CreateDataset_Sketchy(opt, on_Fly=True)
    dataloader_sketchy = data.DataLoader(dataset_sketchy, batch_size=opt.batchsize, shuffle=opt.shuffle,
                                         num_workers=int(opt.nThreads))

    for i_batch, sanpled_batch in enumerate(dataloader_sketchy):
        t0 = time.time()
        if i_batch == 1:
            print(len(sanpled_batch['sketch_img'][0]))
        torchvision.utils.save_image(sanpled_batch['sketch_img'][-1], 'sketch_img.jpg', normalize=True)
        torchvision.utils.save_image(sanpled_batch['positive_img'], 'positive_img.jpg', normalize=True)
        # torchvision.utils.save_image(sanpled_batch['negetive_img'], 'negetive_img.jpg', normalize=True)
        # print(i_batch, sanpled_batch['class_label'], (time.time() - t0))
        # print(sanpled_batch['sketch_img'][-1][0])
        # for i_num in range(len(sanpled_batch['sketch_img'])):
        #    torchvision.utils.save_image(sanpled_batch['sketch_img'][i_num], str(i_num) + 'sketch_img.jpg',
        #                                 normalize=True)
