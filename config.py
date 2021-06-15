class Config(object):
    def __init__(self):
        # self.path = './data/CAMRa2011/'
        self.path = './data/MaFengWo/'
        self.user_dataset = self.path + 'userRating'
        self.group_dataset = self.path + 'groupRating'
        self.user_in_group_path = self.path + 'groupMember.txt'
        self.emb_size = 32
        self.epoch = 200
        self.user_epoch = 10
        self.num_negatives = 10
        self.layers = 2
        self.batch_size = 512
        # self.lr = [0.000005, 0.000001, 0.0000005]
        self.lr = [0.0001, 0.00005, 0.00002]
        self.drop_ratio = 0.1
        self.topK = [5, 10]
        self.balance = 6
        self.gpu_id = 0
