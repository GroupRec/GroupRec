import torch
from torch.autograd import Variable
import numpy as np
import math
import heapq

class Helper(object):
    """
        utils class: it can provide any function that we need
    """
    def __init__(self):
        self.timber = True
    def evaluate_model(self, model, testRatings, testNegatives, device, K_list, type_m):
        """
        Evaluate the performance (Hit_Ratio, NDCG) of top-K recommendation
        Return: score of each test rating.
        """
        hits, ndcgs = [], []
        user_test = []
        item_test = []
        for idx in range(len(testRatings)):
            rating = testRatings[idx]
            items = [rating[1]]
            items.extend(testNegatives[idx])
            item_test.append(items)
            user = np.full(len(items), rating[0])
            user_test.append(user)
        users_var = torch.LongTensor(user_test).to(device)
        items_var = torch.LongTensor(item_test).to(device)

        bsz = len(testRatings)
        item_len = len(testNegatives[0]) + 1

        users_var = users_var.view(-1)
        items_var = items_var.view(-1)
        if type_m == 'group':
            predictions = model(users_var, None, items_var)
        elif type_m == 'user':
            predictions = model(None, users_var, items_var)
        predictions = torch.reshape(predictions, (bsz, item_len))
        pred_score =  predictions.data.cpu().numpy()
        pred_rank = np.argsort(pred_score * -1, axis=1)
        for k in K_list:
            hits.append(getHitK(pred_rank, k))
            ndcgs.append(getNdcgK(pred_rank, k))
        return (hits, ndcgs)

    # def getHitRatio(self, ranklist, gtItem):
    #     for item in ranklist:
    #         if item == gtItem:
    #             return 1
    #     return 0

    # def getNDCG(self, ranklist, gtItem):
    #     for i in range(len(ranklist)):
    #         item = ranklist[i]
    #         if item == gtItem:
    #             return math.log(2) / math.log(i+2)
    #     return 0

def getHitK(pred_rank, k):
    pred_rank_k = pred_rank[:, :k]
    hit = np.count_nonzero(pred_rank_k == 0)
    hit = hit / pred_rank.shape[0]
    return hit

def getNdcgK(pred_rank, k):
    ndcgs = np.zeros(pred_rank.shape[0])
    for user in range(pred_rank.shape[0]):
        for j in range(k):
            if pred_rank[user][j] == 0:
                ndcgs[user] = math.log(2) / math.log(j + 2)
    ndcg = np.mean(ndcgs)
    return ndcg