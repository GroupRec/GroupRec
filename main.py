import torch
import torch.nn as nn
import torch.autograd as autograd
import random
import torch.optim as optim
import numpy as np
from config import Config
from util import Helper
from dataset import GDataset
from model import HGR
from time import time

from tqdm import tqdm

def training(model, train_loader, epoch_id, config, device, type_m):
    learning_rate = config.lr
    lr = learning_rate[0]
    if epoch_id >= 20 and epoch_id < 40:
        lr = learning_rate[0]
    elif epoch_id >= 40:
        lr = learning_rate[1]

    # if epoch_id != 0 and epoch_id % 2 == 0:
    #     lr /= 2

    optimizer = optim.RMSprop(model.parameters(), lr)
    total_loss = []
    # print('%s train_loader length: %d' % (type_m, len(train_loader)))
    for batch_id, (u, pi_ni) in tqdm(enumerate(train_loader)):
        user = torch.LongTensor(u).to(device)
        pos_item = pi_ni[:, 0].to(device)
        neg_item = pi_ni[:, 1].to(device)
        if type_m == 'user':
            pos_predict = model(None, user, pos_item)
            neg_predict = model(None, user, neg_item)
        elif type_m == 'group':
            pos_predict = model(user, None, pos_item)
            neg_predict = model(user, None, neg_item)
        model.zero_grad()
        loss = torch.mean((pos_predict - neg_predict - 1) ** 2)
        total_loss.append(loss)
        loss.backward()
        optimizer.step()

    print('Epoch %d, %s loss is [%.4f]' % (epoch_id, type_m, torch.mean(torch.stack(total_loss))))

def evaluation(model, helper, testRatings, testNegative, device, K_list, type_m):
    model.eval()
    hits, ndcgs = helper.evaluate_model(model, testRatings, testNegative, device, K_list, type_m)
    # hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
    return hits, ndcgs

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed) # cpu
    torch.cuda.manual_seed_all(seed)  # gpu
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    config = Config()
    helper = Helper()

    set_seed(1)

    dataset = GDataset(config.user_dataset, config.group_dataset, config.user_in_group_path, config.num_negatives)

    device_id = "cuda:" + str(config.gpu_id)
    device = torch.device(device_id if torch.cuda.is_available() else "cpu")

    num_users, num_items, num_groups = dataset.num_users, dataset.num_items, dataset.num_groups
    group_member_dict = dataset.group_member_dict

    adj, D, A = dataset.adj, dataset.D, dataset.A
    D = torch.Tensor(D).to(device)
    A = torch.Tensor(A).to(device)
    values = adj.data
    indices = np.vstack((adj.row, adj.col))
    i = torch.LongTensor(indices).to(device)
    v = torch.FloatTensor(values).to(device)
    shape = adj.shape
    adj = torch.sparse.FloatTensor(i, v, torch.Size(shape)).to(device)

    model = HGR(num_users, num_items, num_groups, config.emb_size, config.layers, config.drop_ratio, adj, D, A, group_member_dict, device)
    model = model.to(device)


    for epoch in range(config.epoch):
        model.train()
        t1 = time()
        training(model, dataset.get_group_dataloader(config.batch_size), epoch, config, device, 'group')
        t2 = time()
        training(model, dataset.get_user_dataloader(config.batch_size), epoch, config, device, 'user')
        # torch.save(model, 'model.pkl')
        # model = torch.load('model.pkl')
        hr, ndcg = evaluation(model, helper, dataset.group_testRatings, dataset.group_testNegatives, device, config.topK, 'group')
        print(
            'Group Epoch %d [%.1f s]: \n HR@5 = %.4f, HR@10 = %.4f; '
            'NDCG@5 = %.4f, NDCG@10 = %.4f [%.1f s]' % (epoch, time() - t1, hr[0], hr[1], ndcg[0], ndcg[1], time() - t2))
        for i in range(3):
            u_hr, u_ndcg = evaluation(model, helper, dataset.user_testRatings, dataset.user_testNegatives, device, config.topK, 'user')
        print(
            'User Epoch %d [%.1f s]: \n HR@5 = %.4f, HR@10 = %.4f; '
            'NDCG@5 = %.4f, NDCG@10 = %.4f [%.1f s]\n' % (epoch, time() - t1, u_hr[0], u_hr[1], u_ndcg[0], u_ndcg[1], time() - t2))

    print("Done!")

