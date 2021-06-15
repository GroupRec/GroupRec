'''
Created on Aug 8, 2016
Processing datasets.

@author: Xiangnan He (xiangnanhe@gmail.com)

Modified  on Nov 10, 2017, by Lianhai Miao
'''

import scipy.sparse as sp
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from collections import defaultdict
class GDataset(object):

    def __init__(self, user_path, group_path, user_in_group_path, num_negatives):
        '''
        Constructor
        '''
        self.num_negatives = num_negatives
        # user data
        self.user_trainMatrix = self.load_rating_file_as_matrix(user_path + "Train.txt")
        self.user_testRatings = self.load_rating_file_as_list(user_path + "Test.txt")
        self.user_testNegatives = self.load_negative_file(user_path + "Negative.txt")
        self.num_users, self.num_items = self.user_trainMatrix.shape
        # group data
        self.group_trainMatrix = self.load_rating_file_as_matrix(group_path + "Train.txt")
        self.group_testRatings = self.load_rating_file_as_list(group_path + "Test.txt")
        self.group_testNegatives = self.load_negative_file(group_path + "Negative.txt")
        self.num_groups = self.group_trainMatrix.shape[0]

        self.adj, group_data, self.group_member_dict = self.get_hyper_adj(user_in_group_path, group_path + "Train.txt")
        self.D, self.A = self.get_group_adj(group_data)

    def load_rating_file_as_list(self, filename):
        ratingList = []
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split(" ")
                user, item = int(arr[0]), int(arr[1])
                ratingList.append([user, item])
                line = f.readline()
        return ratingList

    def load_negative_file(self, filename):
        negativeList = []
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split(" ")
                negatives = []
                for x in arr[1:]:
                    negatives.append(int(x))
                negativeList.append(negatives)
                line = f.readline()
        return negativeList

    def load_rating_file_as_matrix(self, filename):
        # Get number of users and items
        num_users = 0
        num_items = 0
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split(" ")
                u, i = int(arr[0]), int(arr[1])
                num_users = max(num_users, u)
                num_items = max(num_items, i)
                line = f.readline()
        # Construct matrix
        mat = sp.dok_matrix((num_users + 1, num_items + 1), dtype=np.float32)
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split(" ")
                if len(arr) > 2:
                    user, item, rating = int(arr[0]), int(arr[1]), int(arr[2])
                    if (rating > 0):
                        mat[user, item] = 1.0
                else:
                    user, item = int(arr[0]), int(arr[1])
                    mat[user, item] = 1.0
                line = f.readline()
        return mat

    def get_hyper_adj(self, user_in_group_path, group_train_path):
        g_m_d = {}
        with open(user_in_group_path, 'r') as f:
            line = f.readline().strip()
            while line != None and line != "":
                a = line.split(' ')
                g = int(a[0])
                g_m_d[g] = []
                for m in a[1].split(','):
                    g_m_d[g].append(int(m))
                line = f.readline().strip()

        g_i_d = defaultdict(list)
        with open(group_train_path, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split(" ")
                if len(arr) > 2:
                    group, item, rating = int(arr[0]), int(arr[1]), int(arr[2])
                    if (rating > 0):
                        g_i_d[group].append(item + self.num_users)
                else:
                    group, item = int(arr[0]), int(arr[1])
                    g_i_d[group].append(item + self.num_users)
                line = f.readline()
        group_data = []
        for i in range(self.num_groups):
            group_data.append(g_m_d[i] + g_i_d[i])

        # group_data = np.array(data)

        # a = [1] * (self.num_users + self.num_items)
        # for item in group_data:
        #     for u in item:
        #         a[u] = a[u] - 1
        # b = []
        # cnt = 0
        # for item in a:
        #     if item == 1:
        #         if cnt < self.num_users:
        #             b.append(cnt)
        #         else:
        #             b.append((cnt - self.num_users))
        #     cnt += 1
        # print(b)
        # print(self.num_users)
        # print(self.num_items)

        def _data_masks(all_group_data):
            indptr, indices, data = [], [], []
            indptr.append(0)
            for j in range(len(all_group_data)):
                single_group = np.unique(np.array(all_group_data[j]))
                length = len(single_group)
                s = indptr[-1]
                indptr.append(s + length)
                for i in range(length):
                    indices.append(single_group[i])
                    data.append(1)
            matrix = sp.csr_matrix((data, indices, indptr), shape=(self.num_groups, self.num_users + self.num_items))
            return matrix

        H_T = _data_masks(group_data)
        # print(H_T)
        BH_T = H_T.T.multiply(1.0/(1.0 + H_T.sum(axis=1).reshape(1, -1)))
        BH_T = BH_T.T
        H = H_T.T
        DH = H.T.multiply(1.0/(1.0 + H.sum(axis=1).reshape(1, -1)))
        DH = DH.T
        DHBH_T = np.dot(DH,BH_T)
        return DHBH_T.tocoo(), group_data, g_m_d


    def get_group_adj(self, group_data):
        matrix = np.zeros((self.num_groups, self.num_groups))
        for i in range(self.num_groups):
            group_a = set(group_data[i])
            for j in range(i + 1, self.num_groups):
                group_b = set(group_data[j])
                overlap = group_a.intersection(group_b)
                ab_set = group_a | group_b
                matrix[i][j] = float(len(overlap) / len(ab_set))
                matrix[j][i] = matrix[i][j]
        matrix = matrix + np.diag([1.0] * self.num_groups)
        degree = np.sum(np.array(matrix), 1)
        degree = np.diag(1.0 / degree)
        return matrix, degree

    def get_train_instances(self, train):
        user_input, pos_item_input, neg_item_input = [], [], []
        num_users = train.shape[0]
        num_items = train.shape[1]
        for (u, i) in train.keys():
            # positive instance
            for _ in range(self.num_negatives):
                pos_item_input.append(i)
            # negative instances
            for _ in range(self.num_negatives):
                j = np.random.randint(num_items)
                while (u, j) in train:
                    j = np.random.randint(num_items)
                user_input.append(u)
                neg_item_input.append(j)
        pi_ni = [[pi, ni] for pi, ni in zip(pos_item_input, neg_item_input)]
        return user_input, pi_ni

    def get_user_dataloader(self, batch_size):
        user, positem_negitem_at_u = self.get_train_instances(self.user_trainMatrix)
        train_data = TensorDataset(torch.LongTensor(user), torch.LongTensor(positem_negitem_at_u))
        user_train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        return user_train_loader

    def get_group_dataloader(self, batch_size):
        group, positem_negitem_at_g = self.get_train_instances(self.group_trainMatrix)
        train_data = TensorDataset(torch.LongTensor(group), torch.LongTensor(positem_negitem_at_g))
        group_train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        return group_train_loader






