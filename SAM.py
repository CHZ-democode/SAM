#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import numpy as np
import scipy.sparse as sp
import torch
from torch import optim
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from tqdm import tqdm
from deeprobust.graph import utils
from deeprobust.graph.global_attack import BaseAttack
import matplotlib.pyplot as plt


class BaseMeta(BaseAttack):

    def __init__(self, model=None, nnodes=None, feature_shape=None, lambda_=0.5, attack_structure=True, attack_features=False, undirected=True, device='cpu'):

        super(BaseMeta, self).__init__(model, nnodes, attack_structure, attack_features, device)
        self.lambda_ = lambda_

        assert attack_features or attack_structure, 'attack_features or attack_structure cannot be both False'

        self.modified_adj = None
        self.modified_features = None

        if attack_structure:
            self.undirected = undirected
            assert nnodes is not None, 'Please give nnodes='
            self.adj_changes = Parameter(torch.FloatTensor(nnodes, nnodes)).to(self.device)
            self.adj_changes.data.fill_(0)

        if attack_features:
            assert feature_shape is not None, 'Please give feature_shape='
            self.feature_changes = Parameter(torch.FloatTensor(feature_shape))
            self.feature_changes.data.fill_(0)

        self.with_relu = model.with_relu

    def attack(self, adj, labels, n_perturbations):
        pass

    def get_modified_adj(self, ori_adj):
        adj_changes_square = self.adj_changes - torch.diag(torch.diag(self.adj_changes, 0))
        # ind = np.diag_indices(self.adj_changes.shape[0]) # this line seems useless
        if self.undirected:
            adj_changes_square = adj_changes_square + torch.transpose(adj_changes_square, 1, 0)
        # print('Here', adj_changes_square.unique())
        adj_changes_square = torch.clamp(adj_changes_square, -1, 1)
        if not torch.is_tensor(ori_adj):
            ori_adj = torch.FloatTensor(ori_adj).to(self.device)

        if not torch.is_tensor(adj_changes_square):
            adj_changes_square = torch.FloatTensor(adj_changes_square).to(self.device)
        adj_changes_square = adj_changes_square.to(self.device)
        modified_adj = adj_changes_square + ori_adj
        return modified_adj

    def get_modified_features(self, ori_features):
        return ori_features + self.feature_changes

    def filter_potential_singletons(self, modified_adj):
        """
        Computes a mask for entries potentially leading to singleton nodes, i.e. one of the two nodes corresponding to
        the entry have degree 1 and there is an edge between the two nodes.
        """

        degrees = modified_adj.sum(0)
        degree_one = (degrees == 1)
        resh = degree_one.repeat(modified_adj.shape[0], 1).float()
        l_and = resh * modified_adj
        if self.undirected:
            l_and = l_and + l_and.t()
        flat_mask = 1 - l_and
        return flat_mask

    def self_training_label(self, labels, idx_train):
        # Predict the labels of the unlabeled nodes to use them for self-training.
        output = self.surrogate.output
        labels_self_training = output.argmax(1)
        labels_self_training[idx_train] = labels[idx_train]
        return labels_self_training


    def get_adj_score(self, adj_grad, modified_adj, ori_adj):
        with torch.no_grad():
            adj_grad = adj_grad.to(self.device)
            adj_meta_grad = adj_grad * (-2 * modified_adj + 1)
            # Make sure that the minimum entry is 0.
            adj_meta_grad -= adj_meta_grad.min()
            # Filter self-loops
            adj_meta_grad -= torch.diag(torch.diag(adj_meta_grad, 0))
            # # Set entries to 0 that could lead to singleton nodes.
            singleton_mask = self.filter_potential_singletons(modified_adj)
            adj_meta_grad = adj_meta_grad *  singleton_mask
        return adj_meta_grad

    def get_feature_score(self, feature_grad, modified_features):
        feature_meta_grad = feature_grad * (-2 * modified_features + 1)
        feature_meta_grad -= feature_meta_grad.min()
        return feature_meta_grad


class SAM(BaseMeta):
    def __init__(self, args, model, nnodes, feature_shape=None, attack_structure=True, attack_features=False, undirected=True, device='cpu', with_bias=False, lambda_=0.5, train_iters=100, lr=0.1, momentum=0.9,
                 node_sim=None, sim_threshold=None, topk=None, dataset=None):

        super(SAM, self).__init__(model, nnodes, feature_shape, lambda_, attack_structure, attack_features, undirected, device)

        # SAM
        self.args = args
        self.dataset = dataset
        self.node_sim = node_sim
        self.topk = topk
        self.alpha_ = Parameter(torch.FloatTensor([0.75])).to(device)
        self.beta_ = Parameter(torch.FloatTensor([0])).to(device)
        self.gamma_ = Parameter(torch.FloatTensor([0.001])).to(device)
        self.huber_ = Parameter(torch.FloatTensor([1])).to(device)
        self.edge_changes_list = []

        self.momentum = momentum
        self.lr = lr
        self.train_iters = train_iters
        self.with_bias = with_bias

        self.weights = []
        self.biases = []
        self.w_velocities = []
        self.b_velocities = []

        self.hidden_sizes = self.surrogate.hidden_sizes
        self.nfeat = self.surrogate.nfeat
        self.nclass = self.surrogate.nclass

        self.loss_list = []
        self.acc_list = []
        self.asr_list = []
        self.attack_loss_list = []

        previous_size = self.nfeat
        for ix, nhid in enumerate(self.hidden_sizes):
            weight = Parameter(torch.FloatTensor(previous_size, nhid).to(device))
            w_velocity = torch.zeros(weight.shape).to(device)
            self.weights.append(weight)
            self.w_velocities.append(w_velocity)

            if self.with_bias:
                bias = Parameter(torch.FloatTensor(nhid).to(device))
                b_velocity = torch.zeros(bias.shape).to(device)
                self.biases.append(bias)
                self.b_velocities.append(b_velocity)

            previous_size = nhid

        output_weight = Parameter(torch.FloatTensor(previous_size, self.nclass).to(device))
        output_w_velocity = torch.zeros(output_weight.shape).to(device)
        self.weights.append(output_weight)
        self.w_velocities.append(output_w_velocity)

        if self.with_bias:
            output_bias = Parameter(torch.FloatTensor(self.nclass).to(device))
            output_b_velocity = torch.zeros(output_bias.shape).to(device)
            self.biases.append(output_bias)
            self.b_velocities.append(output_b_velocity)

        self._initialize()
        #---------------------Calculate similarity matrix----------------------------#
        self.sim_mat = self.get_cosine_similarity_matrix(ori_features_dense)


        sim_mean =(torch.sum(self.sim_mat) - torch.sum(torch.diag(self.sim_mat))) / (self.sim_mat.shape[0] * (self.sim_mat.shape[0] - 1))
        sim_variance = torch.var(self.sim_mat)
        # print('sim_mean', sim_mean)
        # print('sim_variance', sim_variance)

        self.sim_threshold = sim_threshold * sim_variance + sim_mean

        # print('sim_threshold', self.sim_threshold)

        plt.hist(np.load('cosine_sim(2708).npy'))

        plt.legend()
        plt.show()

        # print(sum(sim_mat == 0))
        #---------------------Calculate similarity matrix----------------------------#




    def get_cosine_similarity_matrix(self, features, norm=False):
        sim_mat = np.zeros((features.shape[0], features.shape[0]))

        for i in range(features.shape[0]):
            print(i)
            for j in range(features.shape[0]):
                print(j)
                if i == j:
                    sim_mat[i][j] = float(1)
                else:
                    x = features[i]
                    y = features[j]
                    assert len(x) == len(y), "len(x) != len(y)"
                    zero_list = [0] * len(x)
                    if (x == zero_list).all() or (y == zero_list).all():
                        if x == y:
                            sim_mat[i][j] = float(1)
                        else:
                            sim_mat[i][j] = float(0)
                    else:
                        res = np.array([[x[k] * y[k], x[k] * x[k], y[k] * y[k]] for k in range(len(x))])
                        cos = sum(res[:, 0]) / (np.sqrt(sum(res[:, 1])) * np.sqrt(sum(res[:, 2])))
                        sim_mat[i][j] = 0.5 * cos + 0.5 if norm else cos 
        return sim_mat


    def _initialize(self):
        for w, v in zip(self.weights, self.w_velocities):
            stdv = 1. / math.sqrt(w.size(1))
            w.data.uniform_(-stdv, stdv)
            v.data.fill_(0)

        if self.with_bias:
            for b, v in zip(self.biases, self.b_velocities):
                stdv = 1. / math.sqrt(w.size(1))
                b.data.uniform_(-stdv, stdv)
                v.data.fill_(0)

    def inner_train(self, features, adj_norm, idx_train, idx_unlabeled, labels):
        self._initialize()
        for ix in range(len(self.hidden_sizes) + 1):
            self.weights[ix] = self.weights[ix].detach()
            self.weights[ix].requires_grad = True
            self.w_velocities[ix] = self.w_velocities[ix].detach()
            self.w_velocities[ix].requires_grad = True

            if self.with_bias:
                self.biases[ix] = self.biases[ix].detach()
                self.biases[ix].requires_grad = True
                self.b_velocities[ix] = self.b_velocities[ix].detach()
                self.b_velocities[ix].requires_grad = True

        for j in range(self.train_iters):
            hidden = features
            for ix, w in enumerate(self.weights):
                b = self.biases[ix] if self.with_bias else 0
                if self.sparse_features:
                    hidden = adj_norm @ torch.spmm(hidden, w) + b
                else:
                    hidden = adj_norm @ hidden @ w + b

                if self.with_relu and ix != len(self.weights) - 1:
                    hidden = F.relu(hidden)

            output = F.log_softmax(hidden, dim=1)
            loss_labeled = F.nll_loss(output[idx_train], labels[idx_train])

            weight_grads = torch.autograd.grad(loss_labeled, self.weights, create_graph=True)
            self.w_velocities = [self.momentum * v + g for v, g in zip(self.w_velocities, weight_grads)]
            if self.with_bias:
                bias_grads = torch.autograd.grad(loss_labeled, self.biases, create_graph=True)
                self.b_velocities = [self.momentum * v + g for v, g in zip(self.b_velocities, bias_grads)]

            self.weights = [w - self.lr * v for w, v in zip(self.weights, self.w_velocities)]
            if self.with_bias:
                self.biases = [b - self.lr * v for b, v in zip(self.biases, self.b_velocities)]





    def get_meta_grad(self, features, ori_adj, adj_innorm, adj_norm, idx_train, idx_unlabeled, labels, labels_self_training):

        hidden = features
        hidden_2 = features
        hidden_3 = features.to_dense()


        for ix, w in enumerate(self.weights):
            b = self.biases[ix] if self.with_bias else 0
            if self.sparse_features:
                hidden = adj_norm @ torch.spmm(hidden, w) + b
            else:
                hidden = adj_norm @ hidden @ w + b
            if self.with_relu and ix != len(self.weights) - 1:
                hidden = F.relu(hidden)
        output = F.log_softmax(hidden, dim=1)

        # Output with original adj
        for ix, w in enumerate(self.weights):
            b = self.biases[ix] if self.with_bias else 0
            if self.sparse_features:
                hidden_2 = ori_adj @ torch.spmm(hidden_2, w) + b
            else:
                hidden_2 = ori_adj @ hidden_2 @ w + b
            if self.with_relu and ix != len(self.weights) - 1:
                hidden_2 = F.relu(hidden_2)
        output_ori = F.log_softmax(hidden_2, dim=1)

        loss_labeled = F.nll_loss(output[idx_train], labels[idx_train])
        loss_unlabeled = F.nll_loss(output[idx_unlabeled], labels_self_training[idx_unlabeled])
        loss_test_val = F.nll_loss(output[idx_unlabeled], labels[idx_unlabeled])



        # Loss_sim
        loss_sim = self.feature_smoothing(adj_innorm, hidden_3)
        # Loss_foo
        loss_foo = F.smooth_l1_loss(output[idx_train], output_ori[idx_train], beta=1)
        attack_loss = self.alpha_ * loss_labeled + (1 - self.alpha_) * loss_unlabeled + self.beta_ * loss_foo + self.gamma_ * loss_sim


        self.loss_list.append(loss_test_val.item())
        self.acc_list.append(utils.accuracy(output[idx_unlabeled], labels[idx_unlabeled]).item())
        self.attack_loss_list.append(attack_loss.item())
        print('GCN loss on unlabled data: {}'.format(loss_test_val.item()))
        print('GCN acc on unlabled data: {}'.format(utils.accuracy(output[idx_unlabeled], labels[idx_unlabeled]).item()))
        print('attack loss: {}'.format(attack_loss.item()))

        #---------------------Figure----------------------------#
        # plt.plot(self.loss_list, label="training loss")
        # plt.legend()
        # plt.show()
        #
        # plt.plot(self.attack_loss_list, label="attack loss")
        # plt.legend()
        # plt.show()
        #
        # plt.plot(self.acc_list, label="accuracy")
        # plt.legend()
        # plt.show()
        #---------------------Figure----------------------------#



        #--------------------constraintset---------------------------#
        # alpha
        alpha_grad = None
        alpha_grad = torch.autograd.grad(attack_loss, self.alpha_, create_graph=True, allow_unused=True)[0]
        # print('alpha_grad is ', alpha_grad)
        alpha_grad = alpha_grad.detach()
        self.alpha_ = self.alpha_ - alpha_grad * self.lr
        if not self.args.free_alpha:
            self.alpha_ = self.alpha_.clamp(-1, -0.5)
        # print('alpha is', self.alpha_)

        # beta
        beta_grad = None
        beta_grad = torch.autograd.grad(attack_loss, self.beta_, create_graph=True, allow_unused=True)[0]
        # print('beta_grad is ', beta_grad)
        beta_grad = beta_grad.detach()
        self.beta_ = self.beta_ - beta_grad * self.lr
        if not self.args.free_beta:
            self.beta_ = self.beta_.clamp(-1, 0)
        # print('beta is', self.beta_)


        # gamma
        gamma_grad = None
        gamma_grad = torch.autograd.grad(attack_loss, self.gamma_, create_graph=True, allow_unused=True)[0]
        # print('gamma_grad is ', gamma_grad)
        gamma_grad = gamma_grad.detach()
        self.gamma_ = self.gamma_ - gamma_grad * self.lr
        if not self.args.free_gamma:
            self.gamma_ = self.gamma_.clamp(0.00005, 0.00015)
        # print('gamma is', self.gamma_)





        adj_grad, feature_grad = None, None
        if self.attack_structure:
            adj_grad = torch.autograd.grad(attack_loss, self.adj_changes, retain_graph=True)[0]
        if self.attack_features:
            feature_grad = torch.autograd.grad(attack_loss, self.feature_changes, retain_graph=True)[0]
        return adj_grad, feature_grad

    def feature_smoothing(self, adj, X):
        rowsum = adj.sum(1)
        r_inv = rowsum.flatten()
        D = torch.diag(r_inv)
        L = D - adj

        r_inv = r_inv + 1e-3
        r_inv = r_inv.pow(-1 / 2).flatten()
        r_inv[torch.isinf(r_inv)] = 0.
        r_mat_inv = torch.diag(r_inv)
        #L = r_mat_inv @ L @ r_mat_inv
        L = torch.matmul(torch.matmul(r_mat_inv, L), r_mat_inv)

        XLXT = torch.matmul(torch.matmul(X.t(), L), X)
        loss_smooth_feat = torch.trace(XLXT)
        return loss_smooth_feat

    def attack(self, ori_features, ori_adj, labels, idx_train, idx_unlabeled, n_perturbations):
        self.sparse_features = sp.issparse(ori_features)
        ori_adj, ori_features, labels = utils.to_tensor(ori_adj, ori_features, labels, device=self.device)
        labels_self_training = self.self_training_label(labels, idx_train)
        ori_adj = ori_adj.to_dense()
        ori_adj_norm = utils.normalize_adj_tensor(ori_adj)

        modified_adj = ori_adj
        modified_features = ori_features


        flip_counter = 0
        not_flip_counter = 0
        edge_num = int((torch.sum(ori_adj) // 2 * 0.1).detach().cpu())

        for i in tqdm(range(n_perturbations), desc="Perturbing graph"):
            # Check circuit breaker
            flipped_edge_num = torch.sum(torch.abs(ori_adj - modified_adj)) // 2
            flipped_edge_num = int(flipped_edge_num.detach().cpu())

            if flipped_edge_num > edge_num:
                print('circuit breaker is triggered in the {} round'.format(i))
                print('the num of flipped edge is', flipped_edge_num)
                break

            if self.attack_structure:
                modified_adj = self.get_modified_adj(ori_adj)






            adj_norm = utils.normalize_adj_tensor(modified_adj)
            self.inner_train(modified_features, adj_norm, idx_train, idx_unlabeled, labels)

            adj_grad, feature_grad = self.get_meta_grad(modified_features, ori_adj_norm, modified_adj, adj_norm, idx_train, idx_unlabeled, labels, labels_self_training)

            adj_meta_score = torch.tensor(0.0).to(self.device)
            feature_meta_score = torch.tensor(0.0).to(self.device)
            if self.attack_structure:
                adj_meta_score = self.get_adj_score(adj_grad, modified_adj, ori_adj)
            if self.attack_features:
                feature_meta_score = self.get_feature_score(feature_grad, modified_features)



            if adj_meta_score.max() >= feature_meta_score.max():
                real_k = int(self.topk // math.sqrt(i+1))
                print('real_k', real_k)
                adj_meta_argmax = torch.topk(adj_meta_score.flatten(), real_k).indices
                for k in range(real_k):
                    row_idx, col_idx = utils.unravel_index(adj_meta_argmax[k], ori_adj.shape)
                    if self.node_sim:
                        if self.sim_mat[row_idx][col_idx] > self.sim_threshold and modified_adj[row_idx][col_idx] == 1:  # The larger the more similar
                            not_flip_counter += 1
                            pass
                        else:
                            flip_counter += 1
                            self.adj_changes.data[row_idx][col_idx] += (-2 * modified_adj[row_idx][col_idx] + 1)
                            # assert modified_adj[row_idx][col_idx] == 0
                            if self.undirected:
                                self.adj_changes.data[col_idx][row_idx] += (-2 * modified_adj[row_idx][col_idx] + 1)




            else:
                feature_meta_argmax = torch.argmax(feature_meta_score)
                row_idx, col_idx = utils.unravel_index(feature_meta_argmax, ori_features.shape)
                self.feature_changes.data[row_idx][col_idx] += (-2 * modified_features[row_idx][col_idx] + 1)

        if self.attack_structure:
            self.modified_adj = self.get_modified_adj(ori_adj).detach()
        if self.attack_features:
            self.modified_features = self.get_modified_features(ori_features).detach()

        print('The number of flipped edges', flip_counter, not_flip_counter)



