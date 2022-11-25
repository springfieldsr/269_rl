import  torch.multiprocessing as mp
import time
import torch
import torch.nn.functional as F
import numpy as np
import scipy
import collections

from src.utils.player import Player
from src.utils.utils import *


def logprob2Prob(logprobs,multilabel=False):
    if multilabel:
        probs = torch.sigmoid(logprobs)
    else:
        probs = F.softmax(logprobs, dim=2)
    return probs

def normalizeEntropy(entro,classnum): #this is needed because different number of classes will have different entropy
    maxentro = np.log(float(classnum))
    entro = entro/maxentro  
    return entro

def prob2Logprob(probs,multilabel=False):
    if multilabel:
        raise NotImplementedError("multilabel for prob2Logprob is not implemented")
    else:
        logprobs = torch.log(probs)
    return logprobs


# calculate the percentage of elements smaller than the k-th element
def perc(input, k): return sum([1 if i else 0 for i in input < input[k]]) / float(len(input))


def percd(input):
    # rank = scipy.stats.rankdata(input, method="ordinal")
    # total = len(input)
    # return [(total - rank[i]) / total for i in range(total)]
    total = len(input)
    _, ranking = torch.sort(input)
    ranking = (total - ranking - 1) / total
    return ranking


def degprocess(deg):
    # deg = torch.log(1+deg)
    #return deg/20.
    return torch.clamp_max(deg / 20., 1.)

def feature_similarity(p, mode):
    train_indices = (p.trainmask == 1).nonzero(as_tuple=True)[1].view(p.batchsize, -1)
    if train_indices.nelement() == 0:
        return torch.zeros((p.batchsize, p.G.X.shape[0])).cuda()

    similarity = []
    for idx, row in enumerate(train_indices):
        if mode == "raw_feature":
            matrix = p.G.X
        elif mode == "embedding":
            matrix = p.allnodes_output.transpose(1, 2)[idx]
        else:
            raise NotImplementedError
        train_features = matrix[row, :]
        sim = torch.matmul(matrix, train_features.T)
        sim = torch.max(sim, dim=1)[0]
        sim_score = percd(sim)
        similarity.append(sim_score)

    return torch.vstack(similarity)

def localdiversity(probs,adj,deg):
    indices = adj.coalesce().indices()
    N =adj.size()[0]
    classnum = probs.size()[-1]
    maxentro = np.log(float(classnum))
    edgeprobs = probs[:,indices.transpose(0,1),:]
    headprobs = edgeprobs[:,:,0,:]
    tailprobs = edgeprobs[:,:,1,:]
    kl_ht = (torch.sum(torch.log(torch.clamp_min(tailprobs,1e-10))*tailprobs,dim=-1) - \
        torch.sum(torch.log(torch.clamp_min(headprobs,1e-10))*tailprobs,dim=-1)).transpose(0,1)
    kl_th = (torch.sum(torch.log(torch.clamp_min(headprobs,1e-10))*headprobs,dim=-1) - \
        torch.sum(torch.log(torch.clamp_min(tailprobs,1e-10))*headprobs,dim=-1)).transpose(0,1)
    sparse_output_kl_ht = torch.sparse.FloatTensor(indices,kl_ht,size=torch.Size([N,N,kl_ht.size(-1)]))
    sparse_output_kl_th = torch.sparse.FloatTensor(indices,kl_th,size=torch.Size([N,N,kl_th.size(-1)]))
    sum_kl_ht = torch.sparse.sum(sparse_output_kl_ht,dim=1).to_dense().transpose(0,1)
    sum_kl_th = torch.sparse.sum(sparse_output_kl_th,dim=1).to_dense().transpose(0,1)
    mean_kl_ht = sum_kl_ht/(deg+1e-10)
    mean_kl_th = sum_kl_th/(deg+1e-10)
    # normalize
    mean_kl_ht = mean_kl_ht / mean_kl_ht.max(dim=1, keepdim=True).values
    mean_kl_th = mean_kl_th / mean_kl_th.max(dim=1, keepdim=True).values
    return mean_kl_ht,mean_kl_th


class Env(object):
    ## an environment for multiple players testing the policy at the same time
    def __init__(self,players,args):
        '''
        players: a list containing main player (many task) (or only one task
        '''
        self.players = players
        self.args = args
        self.nplayer = len(self.players)
        self.graphs = [p.G for p in self.players]
        featdim =-1
        self.statedim = self.getState(0).size(featdim)


    def step(self,actions,playerid=0):
        p = self.players[playerid]
        p.query(actions)
        p.trainOnce()
        reward = p.validation(test=False, rerun=False)
        return reward


    def getState(self,playerid=0):
        p = self.players[playerid]
        output = logprob2Prob(p.allnodes_output.transpose(1,2),multilabel=p.G.stat["multilabel"])
        state = self.makeState(output, p)
        return state


    def reset(self,playerid=0):
        self.players[playerid].reset(fix_test=False)

    
    def makeState(self, probs, p, multilabel=False):
        entro = entropy(probs, multilabel=multilabel)
        entro = normalizeEntropy(entro, probs.size(-1)) ## in order to transfer
        deg = degprocess(p.G.deg.expand([probs.size(0)]+list(p.G.deg.size())))
        cent = p.G.centrality
        cent = cent.expand([probs.size(0)]+list(p.G.centrality.size()))

        features = []
        if self.args.use_entropy:
            features.append(entro)
        if self.args.use_degree:
            features.append(deg)
        if self.args.use_local_diversity:
            mean_kl_ht,mean_kl_th = localdiversity(probs, p.G.adj, p.G.deg)
            features.extend([mean_kl_ht, mean_kl_th])
        if self.args.use_select:
            features.append(p.trainmask)
        if self.args.use_centrality:
            features.append(cent)
        if self.args.use_feature_similarity:
            sim = feature_similarity(p, mode="raw_feature")
            features.append(sim)
        if self.args.use_embedding_similarity:
            sim = feature_similarity(p, mode="embedding")
            features.append(sim)

        state = torch.stack(features, dim=-1)
        return state