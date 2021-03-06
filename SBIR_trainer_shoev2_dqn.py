import os
import time
import shutil
import pickle
import numpy as np
import matplotlib.pyplot as plt
import random

import torch
import torch.nn.functional as F
import torch.nn.utils as utils

from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tensorboard_logger import configure, log_value

from model import RecurrentAttention
from utils import AverageMeter


class Trainer:
    """A Recurrent Attention Model trainer.

    All hyperparameters are provided by the user in the
    config file.
    """

    def __init__(self, config, data_loader):
        """
        Construct a new Trainer instance.

        Args:
            config: object containing command line arguments.
            data_loader: A data iterator.
        """
        self.config = config

        if config.use_gpu and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # glimpse network params
        self.patch_size = config.patch_size
        self.glimpse_scale = config.glimpse_scale
        self.num_patches = config.num_patches
        self.loc_hidden = config.loc_hidden
        self.glimpse_hidden = config.glimpse_hidden

        # core network params
        self.num_glimpses = config.num_glimpses
        self.hidden_size = config.hidden_size

        # reinforce params
        self.std = config.std
        self.M = config.M

        # data params
        if config.is_train:
            self.train_loader = data_loader[0]
            self.valid_loader = data_loader[1]
            self.num_train = len(self.train_loader)
            self.num_valid = len(self.valid_loader)
        else:
            self.test_loader = data_loader
            self.num_test = len(self.test_loader)

        self.act_dimension = 64
        self.num_channels = 1

        # training params
        self.epochs = config.epochs
        self.start_epoch = 0
        self.momentum = config.momentum
        self.lr = config.init_lr

        # misc params
        self.best = config.best
        self.ckpt_dir = config.ckpt_dir
        self.logs_dir = config.logs_dir
        self.best_valid_reward = 0.0
        self.counter = 0
        self.lr_patience = config.lr_patience
        self.train_patience = config.train_patience
        self.use_tensorboard = config.use_tensorboard
        self.resume = config.resume
        self.print_freq = config.print_freq
        self.plot_freq = config.plot_freq
        self.model_name = "ram_{}_{}x{}_{}".format(
            config.num_glimpses,
            config.patch_size,
            config.patch_size,
            config.glimpse_scale,
        )

        self.plot_dir = "./plots/" + self.model_name + "/"
        if not os.path.exists(self.plot_dir):
            os.makedirs(self.plot_dir)

        # configure tensorboard logging
        if self.use_tensorboard:
            tensorboard_dir = self.logs_dir + self.model_name
            print("[*] Saving tensorboard logs to {}".format(tensorboard_dir))
            if not os.path.exists(tensorboard_dir):
                os.makedirs(tensorboard_dir)
            configure(tensorboard_dir)

        # build RAM model
        self.model = RecurrentAttention(
            self.patch_size,
            self.num_patches,
            self.glimpse_scale,
            self.num_channels,
            self.loc_hidden,
            self.glimpse_hidden,
            self.std,
            self.hidden_size,
            self.act_dimension,
        )
        self.model.to(self.device)

        # initialize optimizer and scheduler
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.config.init_lr
        )
        self.opt_action = torch.optim.Adam(self.model.policy.corenet_list.parameters(), lr=config.init_lr)
        self.opt_value = torch.optim.Adam(self.model.policy.net_list.parameters(), lr=config.init_lr)

        self.scheduler = ReduceLROnPlateau(
            self.optimizer, "max", factor=0.1, patience=self.lr_patience
        )

        with open("Train.pickle", "rb") as f:
            self.Image_Array_Train, self.Sketch_Array_Train, self.Image_Name_Train, self.Sketch_Name_Train = pickle.load(
                f)
        with open("Test.pickle", "rb") as f:
            self.Image_Array_Test, self.Sketch_Array_Test, self.Image_Name_Test, self.Sketch_Name_Test = pickle.load(f)
        # with open("TrainRL.pickle", "rb") as f:
        #     self.Sketch_Array_Train_RL, self.Sketch_Name_Train_RL = pickle.load(f)
        # with open("TestRL.pickle", "rb") as f:
        #     self.Sketch_Array_Test_RL, self.Sketch_Name_Test_RL = pickle.load(f)
        # self.Sketch_Array_Train_RL = torch.stack(self.Sketch_Array_Train_RL)
        # print(self.Sketch_Array_Train_RL.shape)
        # self.Sketch_Array_Test_RL = torch.stack(self.Sketch_Array_Test_RL)
        # print(self.Sketch_Array_Test_RL.shape)

        print("pretrained load completed!")
        # self.Sketch_Array_Valid = self.Sketch_Array_Test[:100]
        # self.Sketch_Name_Valid = self.Sketch_Name_Test[:100]

    def reset(self):
        h_t = torch.zeros(
            self.batch_size,
            self.hidden_size,
            dtype=torch.float,
            device=self.device,
            requires_grad=True,
        )
        l_t = torch.FloatTensor(self.batch_size, 2).uniform_(-1, 1).to(self.device)
        l_t.requires_grad = True

        return h_t, l_t

    def train(self):
        """Train the model on the training set.

        A checkpoint of the model is saved after each epoch
        and if the validation accuracy is improved upon,
        a separate ckpt is created for use on the test set.
        """
        # load the most recent checkpoint
        if self.resume:
            self.load_checkpoint(best=False)

        print(
            "\n[*] Train on {} samples, validate on {} samples".format(
                self.num_train, self.num_valid
            )
        )

        dict_list = []
        epoches = []
        t_loss = []
        t_reward = []
        v_reward = []
        v_acc = []
        v_acc10 = []
        v_rp = []
        counter1 = 0

        for epoch in range(self.start_epoch, self.epochs):

            print(
                "\nEpoch: {}/{} - LR: {:.6f}".format(
                    epoch + 1, self.epochs, self.optimizer.param_groups[0]["lr"]
                )
            )

            # train for 1 epoch
            train_loss, train_reward, train_loss_action, train_loss_reinforce = self.train_one_epoch(epoch, dict_list,
                                                                                                     counter1)

            # evaluate on validation set
            valid_reward, valid_acc, valid_acc10, rp = self.validate(epoch)

            # reduce lr if validation loss plateaus
            self.scheduler.step(valid_reward)

            is_best = valid_reward > self.best_valid_reward
            msg1 = "train loss: {:.3f} - train reward: {:.3f} - train action_loss: {:.3f} - train reinforce_loss: {:.3f}"
            msg2 = "- val reward: {:.3f} - val acc: {:.3f} - val err: {:.3f}"
            if is_best:
                self.counter = 0
                msg2 += " [*]"
            msg = msg1 + msg2
            print(
                msg.format(
                    train_loss, train_reward, train_loss_action, train_loss_reinforce, valid_reward, valid_acc,
                    1 - valid_acc
                )
            )
            epoches.append(epoch)
            t_loss.append(train_loss)
            t_reward.append(train_reward)
            v_reward.append(valid_reward)
            v_acc.append(valid_acc)
            v_acc10.append(valid_acc10)
            v_rp.append(rp)
            counter1 += 1

            if self.use_tensorboard:
                log_value("train_loss", train_loss, epoch)
                log_value("train_reward", train_reward, epoch)
                # log_value("train_acc", train_acc, epoch)
                log_value("train_loss_action", train_loss_action, epoch)
                log_value("train_loss_reinforce", train_loss_reinforce, epoch)
                log_value("valid reward", valid_reward, epoch)
                log_value("top5 acc", valid_acc, epoch)
                log_value("top10 acc", valid_acc10, epoch)

            # check for improvement
            if not is_best:
                self.counter += 1
            if self.counter > self.train_patience:
                print("[!] No improvement in a while, stopping training.")
                break

            self.best_valid_reward = max(valid_reward, self.best_valid_reward)
            self.save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "model_state": self.model.state_dict(),
                    "optim_state": self.optimizer.state_dict(),
                    "best_valid_acc": self.best_valid_reward,
                },
                is_best,
            )

        with open("Paint.pickle", "wb") as f:
            pickle.dump(v_reward, f)

        plt.plot(epoches, t_loss, color='blue', label='train_loss')
        plt.ylabel('training loss')
        plt.xlabel('epochs')
        # my_x_ticks = np.arange(0, 1000, 50)
        # plt.xticks(my_x_ticks)
        plt.legend(loc='best')
        plt.title('Training Loss Plot')
        plt.savefig('train_loss.eps')
        plt.close()

        plt.plot(epoches, t_reward, color='blue', label='train_reward')
        plt.ylabel('training reward')
        plt.xlabel('epochs')
        # my_x_ticks = np.arange(0, 1000, 50)
        # plt.xticks(my_x_ticks)
        # my_y_ticks = np.arange(0, 1, 0.05)
        # plt.yticks(my_y_ticks)
        plt.legend(loc='best')
        plt.title('Training Reward Plot')
        plt.savefig('train_reward.eps')
        plt.close()

        plt.figure(figsize=(9, 7))
        plt.plot(epoches, v_reward, color='blue', label='valid_reward', linewidth=3)
        plt.ylabel('valid reward', size=20)
        plt.xlabel('epochs', size=20)
        plt.grid()  # ????????????????????????
        plt.xticks(np.arange(0, 101, step=20), fontsize=20)
        plt.yticks(np.arange(0, 0.34, step=0.08), fontsize=20)
        # plt.legend(loc='best')
        # plt.title('Valid Reward Plot')
        plt.savefig('valid_reward.eps')
        plt.close()

        plt.figure(figsize=(9, 7))
        plt.plot(epoches, v_acc, color='blue', label='valid_acc', linewidth=3)
        plt.ylabel('valid accuracy', size=20)
        plt.xlabel('epochs', size=20)
        plt.grid()  # ????????????????????????
        plt.xticks(np.arange(0, 101, step=20), fontsize=20)
        plt.yticks(np.arange(0, 1, step=0.2), fontsize=20)
        # plt.legend(loc='best')
        # plt.title('Valid top5@Accuracy Plot')
        plt.savefig('valid_accuracy.eps')
        plt.close()

        plt.figure(figsize=(9, 7))
        plt.plot(epoches, v_acc10, color='blue', label='valid_acc', linewidth=3)
        plt.ylabel('valid accuracy')
        plt.xlabel('epochs')
        # my_y_ticks = np.arange(0, 1, 0.01)
        # plt.yticks(my_y_ticks)
        plt.legend(loc='best')
        plt.title('Valid top10@Accuracy Plot')
        plt.savefig('valid_accuracy10.eps')
        plt.close()

        plt.plot(epoches, v_rp, color='blue', label='valid_rp')
        plt.ylabel('valid ranking percentile')
        plt.xlabel('epochs')
        # my_y_ticks = np.arange(0, 1, 0.01)
        # plt.yticks(my_y_ticks)
        plt.legend(loc='best')
        plt.title('Valid Rank Percentile Plot')
        plt.savefig('valid_rp.eps')
        plt.close()

    def get_reward(self, action, sketch_name):
        sketch_query_name = '_'.join(sketch_name.split('/')[-1].split('_')[:-1])
        position_query = self.Image_Name_Train.index(sketch_query_name)
        target_distance = F.pairwise_distance(action,
                                              self.Image_Array_Train[position_query].unsqueeze(0))
        distance = F.pairwise_distance(action, self.Image_Array_Train)
        rank = distance.le(target_distance).sum()
        if rank.item() == 0:
            reward = 1. / (rank.item() + 1)
        else:
            reward = 1. / rank.item()
        return reward, rank.item(), self.Image_Array_Train[position_query].unsqueeze(0).to(self.device).detach()

    def fix_value(self):
        for name, param in self.model.policy.named_parameters():
            print(name)
            if name in ['corenet_list']:
                param.requires_grad = True
            elif name in ['net_list']:
                param.requires_grad = False

    def fix_action(self):
        for name, param in self.model.policy.named_parameters():
            print(name)
            if name in ['corenet_list']:
                param.requires_grad = False
            elif name in ['net_list']:
                param.requires_grad = True

    def fix_all(self):
        for name, param in self.model.policy.named_parameters():
            print(name)
            param.requires_grad = False

    def get_action(self, q_value, h_t, k, epoch, t, is_test):
        loss_q = -torch.mean(q_value.squeeze(1))
        patience = 10
        counter = 0
        min_loss = loss_q.item()

        self.opt_action.zero_grad()
        loss_q.backward()
        utils.clip_grad_norm_(self.model.policy.corenet_list.parameters(), 40)
        self.opt_action.step()

        eps = random.random()
        if epoch <= 20:
            if eps > 0.5:
                action = torch.FloatTensor(self.batch_size, 2).uniform_(-1, 1).to(self.device)
            else:
                action, _ = self.model.policy(h_t, k, epoch, t, is_test)
        else:
            if eps > 0.8:
                action = torch.FloatTensor(self.batch_size, 2).uniform_(-1, 1).to(self.device)
            else:
                action, _ = self.model.policy(h_t, k, epoch, t, is_test)

        return action.detach(), min_loss

    def pt_train(self, h_t_prev, l_t, r_t, h_t, masks, epoch, t, is_test, n_ensemble=6):
        # min history to learn is 200,000 frames in dqn - 50000 steps
        losses = [0.0 for _ in range(n_ensemble)]
        self.opt_value.zero_grad()

        cnt_losses = []
        for k in range(n_ensemble):
            # TODO finish masking
            total_used = torch.sum(masks[:, k]).item()
            if total_used > 0.0:
                _, next_q_policy_vals = self.model.policy(h_t, k, epoch, t, is_test)
                next_actions, _ = self.get_action(next_q_policy_vals, h_t, k, epoch, t, False)
                _, next_q_target_vals = self.model.target(h_t, k, epoch, t, is_test, next_actions)

                _, q_policy_vals = self.model.policy(h_t_prev, k, epoch, t, is_test, l_t)
                preds = q_policy_vals.squeeze(1)
                if t < 5:
                    targets = r_t + next_q_target_vals
                elif t == 5:
                    targets = r_t
                targets = targets.squeeze(1)
                l1loss = F.smooth_l1_loss(preds, targets.detach())
                full_loss = masks[:, k] * l1loss
                loss = torch.sum(full_loss / total_used)
                cnt_losses.append(loss)
                losses[k] = loss.cpu().detach().item()

        loss = sum(cnt_losses) / n_ensemble
        loss.backward()

        utils.clip_grad_norm_(self.model.policy.net_list.parameters(), 40)
        self.opt_value.step()
        return np.mean(losses)

    def train_one_epoch(self, epoch, dict_list, counter, n_ensemble=6):
        """
        Train the model for 1 epoch of the training set.

        An epoch corresponds to one full pass through the entire
        training set in successive mini-batches.

        This is used by train() and should not be called manually.
        """

        self.model.train()
        batch_time = AverageMeter()
        reward = AverageMeter()
        losses_action = AverageMeter()
        losses_reinforce = AverageMeter()
        # losses_baseline = AverageMeter()
        losses = AverageMeter()
        # accs = AverageMeter()

        tic = time.time()
        # imgs = []
        # locs = []
        with tqdm(total=self.num_train) as pbar:
            for i, sampled_batch in enumerate(self.train_loader):

                self.optimizer.zero_grad()
                plot = False
                if (epoch % self.plot_freq == 0) and (i == 0):
                    plot = True

                imgs = []
                locs_list = []
                loss_buffer = []
                for j, sampled_sketch in enumerate(sampled_batch['sketch_img']):
                    if (epoch == 0 or counter == 0) and i == 0:
                        dict = {}
                        dict_list.append(dict)

                    # x, y = x.to(self.device), y.to(self.device)
                    x = sampled_sketch.to(self.device)

                    # initialize location vector and hidden state
                    self.batch_size = x.shape[0]

                    # h_t = torch.tensor(self.Sketch_Array_Train[i][-1], dtype=torch.float, device=self.device,requires_grad=True).unsqueeze(0)

                    if (i + 1) < len(self.train_loader):
                        h_t = torch.stack(self.Sketch_Array_Train[(i * 128):(i + 1) * 128]).clone().detach().float().to(
                            self.device).requires_grad_(True)
                        h_t = h_t[:, j]
                        standard = h_t
                    else:
                        h_t = torch.stack(self.Sketch_Array_Train[(i * 128):]).clone().detach().float().to(
                            self.device).requires_grad_(True)
                        h_t = h_t[:, j]
                        standard = h_t

                    h_t, l_t = self.reset()

                    # save images
                    if j == 8 or j == 16:
                        imgs.append(x[0:9])
                    # imgs = []
                    # imgs.append(x[0:9])

                    # extract the glimpses
                    locs = []
                    log_pi_ensmeble = []
                    entropys = []

                    np.set_printoptions(threshold=np.inf)
                    random_state = np.random.RandomState(393)
                    masks = np.empty((self.batch_size, n_ensemble), dtype=np.bool)
                    for k in range(self.batch_size):
                        masks[k] = random_state.binomial(1, 0.5, 6)
                    masks = torch.FloatTensor(masks.astype(np.int)).to('cuda')
                    heads = list(range(n_ensemble))
                    random_state.shuffle(heads)
                    active_head = heads[0]
                    for t in range(self.num_glimpses - 1):
                        h_t_prev = h_t
                        # forward pass through model
                        h_t, l_t, q_value = self.model(x, l_t, h_t_prev, epoch, t, False, standard, active_head)
                        l_t, loss_q = self.get_action(q_value, h_t_prev.detach(), active_head, epoch, t, False)
                        # min_loss_q = loss_q
                        # patience = 10
                        # counter = 0
                        # while(counter<patience):
                        #     h_t, l_t, q_value = self.model(x, l_t, h_t_prev, epoch, t, False, standard, active_head)
                        #     l_t, loss_q = self.get_action(q_value, h_t_prev.detach(), active_head, epoch, t, False)
                        #     if(loss_q >= min_loss_q):
                        #         counter += 1
                        #     else:
                        #         min_loss_q = loss_q
                        #         counter = 0

                        print("l_t_{}/{}/{}/{}".format(epoch, i, j, t), l_t.detach())
                        r_t = 0
                        loss_reinforce = self.pt_train(h_t_prev.detach(), l_t, r_t, h_t, masks, epoch, t, False)
                        losses_reinforce.update(loss_reinforce, x.size()[0])

                        # store
                        locs.append(l_t[0:9])

                    # last iteration
                    h_t_prev = h_t
                    h_t, l_t, q_value, action_mean = self.model(x, l_t, h_t_prev, epoch, t, False, standard,
                                                                active_head, last=True)
                    l_t, loss_q = self.get_action(q_value, h_t_prev.detach(), active_head, epoch, t, False)

                    # min_loss_q = loss_q
                    # patience = 10
                    # counter = 0
                    # while (counter < patience):
                    #     h_t, l_t, q_value, action_mean = self.model(x, l_t, h_t_prev, epoch, t, False, standard, active_head, last=True)
                    #     l_t, loss_q = self.get_action(q_value, h_t_prev.detach(), active_head, epoch, t, False)
                    #     if (loss_q >= min_loss_q):
                    #         counter += 1
                    #     else:
                    #         min_loss_q = loss_q
                    #         counter = 0

                    # compute losses for differentiable modules
                    sketch_name_list = sampled_batch['sketch_path']
                    one_hot = []
                    R = []
                    Reward_back = []

                    for k, sketch_name in enumerate(sketch_name_list):
                        assert sketch_name == self.Sketch_Name_Train[i * 128 + k]

                        action_single = action_mean[k].unsqueeze(0)
                        Reward, rank, target_img = self.get_reward(action_single, sketch_name)

                        flag = False
                        if sketch_name in dict_list[j]:
                            if rank <= dict_list[j][sketch_name]:
                                flag = True
                                dict_list[j][sketch_name] = rank
                        else:
                            dict_list[j][sketch_name] = rank
                            flag = True
                        # if k1 == self.num_glimpses - 1:
                        one_hot.append(target_img)

                        if flag:
                            R_back = Reward
                            Reward = 1.
                        else:
                            R_back = Reward
                            Reward = 0.

                        R.append(torch.tensor([Reward]))
                        Reward_back.append(torch.tensor([R_back]))

                    R = torch.stack(R).to(self.device)
                    Reward_back = torch.stack(Reward_back)
                    Reward_back = Reward_back.squeeze().mean()

                    r_t = R
                    loss_reinforce = self.pt_train(h_t_prev.detach(), l_t, r_t, h_t, masks, epoch, t, False)
                    losses_reinforce.update(loss_reinforce, x.size()[0])

                    print("h_t_{}/{}/{}/{}".format(epoch, i, j, t), h_t[0].detach().cpu().numpy())

                    self.model.target.load_state_dict(self.model.policy.state_dict())

                    locs.append(l_t[0:9])

                    if (j == 8 or j == 16):
                        locs_list.append(locs)

                    one_hot = torch.cat(one_hot)

                    loss_action = F.mse_loss(action_mean, one_hot)

                    loss = loss_action
                    loss_buffer.append(loss)

                    losses.update(loss.item(), x.size()[0])
                    losses_action.update(loss_action.item(), x.size()[0])

                    # accs.update(acc.item(), x.size()[0])
                    reward.update(Reward_back, x.size()[0])

                # compute gradients and update SGD
                policy_loss = torch.stack(loss_buffer).mean()
                policy_loss.backward()
                self.optimizer.step()

                # measure elapsed time
                toc = time.time()
                batch_time.update(toc - tic)

                pbar.set_description(
                    (
                        "{:.1f}s - loss: {:.3f} - reward: {:.3f}-action loss: {:.3f}-location loss: {:.3f}".format(
                            (toc - tic), losses.avg, reward.avg, losses_action.avg, losses_reinforce.avg
                        )
                    )
                )
                pbar.update(self.batch_size * 17)

                # dump the glimpses and locs
                if plot:
                    # imgs = [g.cpu().data.numpy().squeeze(1) for g in imgs]
                    # locs = torch.stack(locs).transpose(1,0)
                    # locs = [l.cpu().data.numpy() for l in locs]
                    imgs = torch.cat(imgs).cpu().data.numpy().squeeze(1)
                    locs = []
                    for loc_index in range(6):
                        locs.append(torch.cat([locs_list[0][loc_index], locs_list[1][loc_index]]))
                    # imgs = [g.cpu().data.numpy().squeeze() for g in imgs]
                    # locs = [l.cpu().data.numpy() for l in locs_list[0]]
                    assert len(locs) == self.num_glimpses
                    pickle.dump(
                        imgs, open(self.plot_dir + "g_{}.p".format(epoch + 1), "wb")
                    )
                    pickle.dump(
                        locs, open(self.plot_dir + "l_{}.p".format(epoch + 1), "wb")
                    )

                # log to tensorboard
                # if self.use_tensorboard:
                # iteration = epoch * len(self.train_loader) + i
                # log_value("train_loss", losses.avg, iteration)
                # log_value("train_reward", reward.avg, iteration)
                # log_value("train_acc", accs.avg, iteration)
                # log_value("train_loss_action", losses_action.avg, iteration)
                # log_value("train_loss_reinforce", losses_reinforce.avg, iteration)
                # log_value("train_loss_baseline", losses_baseline.avg, iteration)
                # log_value("action_log_probablity", ac_p,iteration)
                # log_value("location_log_probablity", p, iteration)
                # log_value("h_t_norm", torch.norm(h_t), iteration)

            return losses.avg, reward.avg, losses_action.avg, losses_reinforce.avg

    @torch.no_grad()
    def validate1(self, epoch):
        """Evaluate the RAM model on the validation set.
        """
        # losses = AverageMeter()
        # accs = AverageMeter()
        self.model.eval()
        # num_of_Sketch_Step = len(self.Sketch_Array_Valid[0])
        avererage_area = []
        rank_all = torch.zeros(len(self.Sketch_Array_Valid))
        Image_Array_Valid = []
        previous_query = ""

        for i, sampled_batch in enumerate(self.valid_loader):
            # x, y = x.to(self.device), y.to(self.device)

            sketch_name = self.Sketch_Name_Valid[i]
            assert sketch_name == sampled_batch["sketch_path"][0]
            sketch_query_name = '_'.join(sketch_name.split('/')[-1].split('_')[:-1])
            position_query = self.Image_Name_Test.index(sketch_query_name)
            if (previous_query != position_query):
                previous_query = position_query
                target = self.Image_Array_Test[position_query].unsqueeze(0)
                Image_Array_Valid.append(target)
        Image_Array_Valid = torch.cat(Image_Array_Valid)

        for i, sampled_batch in enumerate(self.valid_loader):

            sketch_name = self.Sketch_Name_Valid[i]
            assert sketch_name == sampled_batch["sketch_path"][0]
            sketch_query_name = '_'.join(sketch_name.split('/')[-1].split('_')[:-1])
            position_query = self.Image_Name_Test.index(sketch_query_name)
            x = sampled_batch['sketch_img'][-1].to(self.device)
            # duplicate M times
            # x = x.repeat(self.M, 1, 1, 1)

            # initialize location vector and hidden state
            self.batch_size = x.shape[0]
            h_t, l_t = self.reset()
            # h_t = torch.tensor(self.Sketch_Array_Valid[i][-1], dtype=torch.float,device=self.device,requires_grad=True ).unsqueeze(0)
            # h_t = self.Sketch_Array_Train[i][-1].clone().detach().float().unsqueeze(0).to(self.device).requires_grad_(True)
            # extract the glimpses
            # log_pi = []
            # baselines = []
            for t in range(self.num_glimpses - 1):
                # forward pass through model
                h_t, l_t, b_t, p, entropy = self.model(x, l_t, h_t, epoch, t, True)

            # last iteration
            h_t, l_t, b_t, action, a_p, p, entropy = self.model(x, l_t, h_t, epoch, t, True, last=True)

            # action, _ = self.model(h_t, last=True)
            target_distance = F.pairwise_distance(action,
                                                  self.Image_Array_Test[position_query].unsqueeze(0))
            distance = F.pairwise_distance(action, Image_Array_Valid)
            rank_all[i] = distance.le(target_distance).sum()

            if rank_all[i].item() == 0:
                avererage_area.append(1.)
            else:
                avererage_area.append(1. / rank_all[i].item())

        top1_accuracy = rank_all.le(5).sum().numpy() / rank_all.shape[0]
        meanIOU = np.mean(avererage_area)
        # log to tensorboard
        if self.use_tensorboard:
            iteration = epoch * len(self.valid_loader) + i
            log_value("valid_avg_reward", meanIOU, iteration)
            log_value("valid_top5_acc", top1_accuracy, iteration)

        return meanIOU, top1_accuracy

    def validate(self, epoch):
        """Evaluate the RAM model on the validation set.
        """

        # num_of_Sketch_Step = len(self.Sketch_Array_Valid[0])
        self.model.eval()
        avererage_area = []
        average_rp = []
        rank_all = torch.zeros(len(self.Sketch_Array_Test), 17)
        # Image_Array_Valid = []
        # previous_query = ""
        #
        # for i, sampled_batch in enumerate(self.valid_loader):
        #     # x, y = x.to(self.device), y.to(self.device)
        #
        #     sketch_name = self.Sketch_Name_Valid[i]
        #     assert sketch_name == sampled_batch["sketch_path"][0]
        #     sketch_query_name = '_'.join(sketch_name.split('/')[-1].split('_')[:-1])
        #     position_query = self.Image_Name_Test.index(sketch_query_name)
        #     if (previous_query != position_query):
        #         previous_query = position_query
        #         target = self.Image_Array_Test[position_query].unsqueeze(0)
        #         Image_Array_Valid.append(target)
        # Image_Array_Valid = torch.cat(Image_Array_Valid)

        for i, sampled_batch in enumerate(self.valid_loader):

            sketch_name = self.Sketch_Name_Test[i]
            assert sketch_name == sampled_batch["sketch_path"][0]
            sketch_query_name = '_'.join(sketch_name.split('/')[-1].split('_')[:-1])
            position_query = self.Image_Name_Test.index(sketch_query_name)

            for j, sampled_sketch in enumerate(sampled_batch['sketch_img']):
                x = sampled_sketch.to(self.device)

                # initialize location vector and hidden state
                self.batch_size = x.shape[0]

                h_t = self.Sketch_Array_Test[i][j].unsqueeze(0).clone().detach().float().to(
                    self.device).requires_grad_(True)

                # h_t = F.normalize(h_t)
                standard = h_t

                # standard1 = self.Sketch_Array_Test_RL[i, j].unsqueeze(0).clone().detach().float().to(
                #     self.device).requires_grad_(True)
                # extract the glimpses
                h_t, l_t = self.reset()

                self.model.train()
                random_state = np.random.RandomState(393)
                heads = list(range(6))
                random_state.shuffle(heads)
                active_head = heads[0]
                for t in range(self.num_glimpses - 1):
                    # forward pass through model
                    h_t_prev = h_t
                    # forward pass through model
                    h_t, l_t, q_value = self.model(x, l_t, h_t_prev, epoch, t, False, standard, active_head)
                    l_t, loss_q = self.get_action(q_value, h_t_prev.detach(), active_head, epoch, t, False)
                    # min_loss_q = loss_q
                    # patience = 10
                    # counter = 0
                    # while (counter < patience):
                    #     h_t, l_t, q_value = self.model(x, l_t, h_t_prev, epoch, t, False, standard, active_head)
                    #     l_t, loss_q = self.get_action(q_value, h_t_prev.detach(), active_head, epoch, t, False)
                    #     if (loss_q >= min_loss_q):
                    #         counter += 1
                    #     else:
                    #         min_loss_q = loss_q
                    #         counter = 0

                # last iteration
                h_t_prev = h_t
                # forward pass through model
                h_t, l_t, q_value, action = self.model(x, l_t, h_t_prev, epoch, t, False, standard, active_head,
                                                       last=True)
                # l_t, loss_q = self.get_action(q_value, h_t_prev.detach(), active_head, epoch, t, False)
                # min_loss_q = loss_q
                # patience = 10
                # counter = 0
                # while (counter < patience):
                #     h_t, l_t, q_value, action = self.model(x, l_t, h_t_prev, epoch, t, False, standard, active_head, last=True)
                #     l_t, loss_q = self.get_action(q_value, h_t_prev.detach(), active_head, epoch, t, False)
                #     if (loss_q >= min_loss_q):
                #         counter += 1
                #     else:
                #         min_loss_q = loss_q
                #         counter = 0

                self.model.eval()
                target_distance = F.pairwise_distance(action,
                                                      self.Image_Array_Test[position_query].unsqueeze(0))

                distance = F.pairwise_distance(action, self.Image_Array_Test)
                rank_all[i, j] = distance.le(target_distance).sum()
                rank_percentile = ((len((distance == target_distance).nonzero(as_tuple=True)) // 2) + len(
                    self.Image_Name_Test) - rank_all[i, j]) / len(self.Image_Name_Test)
                average_rp.append(rank_percentile)

                if rank_all[i, j].item() == 0:
                    avererage_area.append(1.)
                else:
                    avererage_area.append(1. / rank_all[i, j].item())

        top5_accuracy = rank_all[:, -1].le(5).sum().numpy() / rank_all.shape[0]
        top10_accuracy = rank_all[:, -1].le(10).sum().numpy() / rank_all.shape[0]
        meanIOU = np.mean(avererage_area)
        rp = np.mean(average_rp)
        # log to tensorboard
        # if self.use_tensorboard:
        #     iteration = epoch * len(self.valid_loader) + i
        #     log_value("valid_avg_reward", meanIOU, iteration)
        #     log_value("valid_top5_acc", top5_accuracy, iteration)

        self.model.train()

        return meanIOU, top5_accuracy, top10_accuracy, rp

    @torch.no_grad()
    def test1(self):
        """Test the RAM model.

        This function should only be called at the very
        end once the model has finished training.
        """

        # load the best checkpoint
        self.load_checkpoint(best=self.best)
        # self.Sketch_Array_Test = self.Sketch_Array_Test[100:]
        # self.Sketch_Name_Test = self.Sketch_Name_Test[100:]

        self.model.eval()
        avererage_area = []
        average_rp = []
        rank_all = torch.zeros(len(self.Sketch_Array_Test), 17)
        rank_inverse = torch.zeros(len(self.Sketch_Array_Test), 17)
        rank_rp = torch.zeros(len(self.Sketch_Array_Test), 17)
        imgs = []
        locs_list = []
        # Image_Array_Valid = []
        # previous_query = ""
        #
        # for i, sampled_batch in enumerate(self.valid_loader):
        #     # x, y = x.to(self.device), y.to(self.device)
        #
        #     sketch_name = self.Sketch_Name_Valid[i]
        #     assert sketch_name == sampled_batch["sketch_path"][0]
        #     sketch_query_name = '_'.join(sketch_name.split('/')[-1].split('_')[:-1])
        #     position_query = self.Image_Name_Test.index(sketch_query_name)
        #     if (previous_query != position_query):
        #         previous_query = position_query
        #         target = self.Image_Array_Test[position_query].unsqueeze(0)
        #         Image_Array_Valid.append(target)
        # Image_Array_Valid = torch.cat(Image_Array_Valid)

        for i, sampled_batch in enumerate(self.test_loader):

            sketch_name = self.Sketch_Name_Test[i]
            assert sketch_name == sampled_batch["sketch_path"][0]
            sketch_query_name = '_'.join(sketch_name.split('/')[-1].split('_')[:-1])
            position_query = self.Image_Name_Test.index(sketch_query_name)
            if i % 5 == 0:
                imgs = []
                locs_list = []

            for j, sampled_sketch in enumerate(sampled_batch['sketch_img']):
                x = sampled_sketch.to(self.device)

                # initialize location vector and hidden state
                self.batch_size = x.shape[0]
                h_t, l_t = self.reset()
                if j == 8 or j == 16:
                    imgs.append(x)
                # h_t = torch.tensor(self.Sketch_Array_Valid[i][-1],dtype=torch.float,device=self.device,requires_grad=True ).unsqueeze(0)
                # h_t = self.Sketch_Array_Train[i][-1].clone().detach().float().unsqueeze(0).to(self.device).requires_grad_(True)
                # extract the glimpses
                locs = []
                for t in range(self.num_glimpses - 1):
                    # forward pass through model
                    h_t, l_t, b_t, p, entropy = self.model(x, l_t, h_t, 0, t, True)
                    locs.append(p)

                # last iteration
                h_t, l_t, b_t, action, a_p, p, entropy = self.model(x, l_t, h_t, 0, t, True, last=True)

                target_distance = F.pairwise_distance(action,
                                                      self.Image_Array_Test[position_query].unsqueeze(0))
                locs.append(p)
                if j == 8 or j == 16:
                    locs_list.append(locs)
                distance = F.pairwise_distance(action, self.Image_Array_Test)
                rank_all[i, j] = distance.le(target_distance).sum()
                rank_percentile = ((len((distance == target_distance).nonzero(as_tuple=True)) // 2) + len(
                    self.Image_Name_Test) - rank_all[i, j]) / len(self.Image_Name_Test)
                average_rp.append(rank_percentile)
                rank_rp[i, j] = rank_percentile

                if rank_all[i, j].item() == 0:
                    avererage_area.append(1.)
                    rank_inverse[i, j] = 1.
                else:
                    avererage_area.append(1. / rank_all[i, j].item())
                    rank_inverse[i, j] = 1. / rank_all[i, j].item()

            if i % 5 == 4:
                imgs = [g.cpu().data.numpy().squeeze(1) for g in imgs]
                imgs = np.concatenate(imgs)
                locs = []
                for loc_index in range(12):
                    locs.append(torch.cat([locs_list[0][loc_index], locs_list[1][loc_index],
                                           locs_list[2][loc_index], locs_list[3][loc_index],
                                           locs_list[4][loc_index], locs_list[5][loc_index],
                                           locs_list[6][loc_index], locs_list[7][loc_index],
                                           locs_list[8][loc_index], locs_list[9][loc_index]]))

                assert len(locs) == self.num_glimpses
                pickle.dump(
                    imgs, open(self.plot_dir + "test/" + "g_{}.p".format(i + 1), "wb")
                )
                pickle.dump(
                    locs, open(self.plot_dir + "test/" + "l_{}.p".format(i + 1), "wb")
                )

        top5_accuracy = rank_all[:, -1].le(5).sum().numpy() / rank_all.shape[0]
        top10_accuracy = rank_all[:, -1].le(10).sum().numpy() / rank_all.shape[0]
        meanIOU = np.mean(avererage_area)
        rp = np.mean(average_rp)

        rank_inverse = [rank_inverse[:, i].mean() for i in range(17)]
        rank_rp = [rank_rp[:, i].mean() for i in range(17)]

        plt.plot(range(17), rank_inverse, color='blue', label='rank_inv')
        plt.ylabel('test rank inverse')
        plt.xlabel('complete degree')
        # my_y_ticks = np.arange(0, 1, 0.01)
        # plt.yticks(my_y_ticks)
        plt.legend(loc='best')
        plt.title('test rank inverse versus partial sketch')
        plt.savefig('test_rinv.eps')
        plt.close()

        plt.plot(range(17), rank_rp, color='blue', label='rank_rp')
        plt.ylabel('test rank percentile')
        plt.xlabel('complete degree')
        # my_y_ticks = np.arange(0, 1, 0.01)
        # plt.yticks(my_y_ticks)
        plt.legend(loc='best')
        plt.title('test rp inverse versus partial sketch')
        plt.savefig('test_rp.eps')
        plt.close()

        print(
            "[*] MeanIou Test top5_acc Test top10_acc rp: ({:.2f} - {:.2f} - {:.2f} - {:.2f})".format(
                meanIOU, top5_accuracy, top10_accuracy, rp
            )
        )

        return meanIOU, top5_accuracy

    @torch.no_grad()
    def test(self):
        """Test the RAM model.

        This function should only be called at the very
        end once the model has finished training.
        """
        correct = 0

        # load the best checkpoint
        self.load_checkpoint(best=self.best)

        for i, (x, y) in enumerate(self.test_loader):
            x, y = x.to(self.device), y.to(self.device)

            # duplicate M times
            x = x.repeat(self.M, 1, 1, 1)

            # initialize location vector and hidden state
            self.batch_size = x.shape[0]
            h_t, l_t = self.reset()

            # extract the glimpses
            for t in range(self.num_glimpses - 1):
                # forward pass through model
                h_t, l_t, b_t, p = self.model(x, l_t, h_t)

            # last iteration
            h_t, l_t, b_t, log_probas, p = self.model(x, l_t, h_t, last=True)

            log_probas = log_probas.view(self.M, -1, log_probas.shape[-1])
            log_probas = torch.mean(log_probas, dim=0)

            pred = log_probas.data.max(1, keepdim=True)[1]
            correct += pred.eq(y.data.view_as(pred)).cpu().sum()

        perc = (100.0 * correct) / (self.num_test)
        error = 100 - perc
        print(
            "[*] Test Acc: {}/{} ({:.2f}% - {:.2f}%)".format(
                correct, self.num_test, perc, error
            )
        )

    def save_checkpoint(self, state, is_best):
        """Saves a checkpoint of the model.

        If this model has reached the best validation accuracy thus
        far, a seperate file with the suffix `best` is created.
        """
        filename = self.model_name + "_ckpt.pth.tar"
        ckpt_path = os.path.join(self.ckpt_dir, filename)
        torch.save(state, ckpt_path)
        if is_best:
            filename = self.model_name + "_model_best.pth.tar"
            shutil.copyfile(ckpt_path, os.path.join(self.ckpt_dir, filename))

    def load_checkpoint(self, best=False):
        """Load the best copy of a model.

        This is useful for 2 cases:
        - Resuming training with the most recent model checkpoint.
        - Loading the best validation model to evaluate on the test data.

        Args:
            best: if set to True, loads the best model.
                Use this if you want to evaluate your model
                on the test data. Else, set to False in which
                case the most recent version of the checkpoint
                is used.
        """
        print("[*] Loading model from {}".format(self.ckpt_dir))

        filename = self.model_name + "_ckpt.pth.tar"
        if best:
            filename = self.model_name + "_model_best.pth.tar"
        ckpt_path = os.path.join(self.ckpt_dir, filename)
        ckpt = torch.load(ckpt_path)

        # load variables from checkpoint
        self.start_epoch = ckpt["epoch"]
        self.best_valid_acc = ckpt["best_valid_acc"]
        self.best_valid_reward = ckpt["best_valid_acc"]
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optim_state"])

        if best:
            print(
                "[*] Loaded {} checkpoint @ epoch {} "
                "with best valid acc of {:.3f}".format(
                    filename, ckpt["epoch"], ckpt["best_valid_acc"]
                )
            )
        else:
            print("[*] Loaded {} checkpoint @ epoch {}".format(filename, ckpt["epoch"]))
