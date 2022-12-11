import numpy as np
import torch
import argparse
from pprint import pformat
from torch.distributions import Categorical
import torch.nn.functional as F
from src.utils.dataloader import GraphLoader
from src.utils.player import Player
from src.utils.env import Env
from src.utils.policynet import *
from src.utils.rewardshaper import RewardShaper
from src.utils.common import *
from src.utils.utils import *
from src.utils.const import MIN_EPSILON


switcher = {'gcn':PolicyNet,'mlp':PolicyNet2}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nhid",type=int,default=64)
    parser.add_argument("--pnhid",type=str,default='8+8')
    parser.add_argument("--dropout",type=float,default=0.2)
    parser.add_argument("--pdropout",type=float,default=0.0)
    parser.add_argument("--lr",type=float,default=3e-2)
    parser.add_argument("--rllr",type=float,default=1e-2)
    parser.add_argument("--entcoef",type=float,default=0)
    parser.add_argument("--frweight",type=float,default=1e-3)
    parser.add_argument("--batchsize",type=int,default=10)
    parser.add_argument("--budgets",type=str,default="35",help="budget per class")
    parser.add_argument("--ntest",type=int,default=1000)
    parser.add_argument("--nval",type=int,default=500)
    parser.add_argument("--datasets",type=str,default="cora")
    parser.add_argument("--metric",type=str,default="microf1")
    parser.add_argument("--remain_epoch",type=int,default=35,help="continues training $remain_epoch"
                                                                  " epochs after all the selection")
    parser.add_argument("--shaping",type=str,default="234",help="reward shaping method, 0 for no shaping;"
                                                              "1 for add future reward,i.e. R= r+R*gamma;"
                                                              "2 for use finalreward;"
                                                              "3 for subtract baseline(value of curent state)"
                                                              "1234 means all the method is used,")
    parser.add_argument("--logfreq",type=int,default=10)
    parser.add_argument("--maxepisode",type=int,default=20000)
    parser.add_argument("--save",type=int,default=0)
    parser.add_argument("--savename",type=str,default="tmp")
    parser.add_argument("--policynet",type=str,default='gcn')
    parser.add_argument("--multigraphindex", type=int, default=0)

    parser.add_argument("--use_entropy",type=int,default=1)
    parser.add_argument("--use_degree",type=int,default=1)
    parser.add_argument("--use_local_diversity",type=int,default=1)
    parser.add_argument("--use_select",type=int,default=1)
    parser.add_argument("--use_centrality", type=int, default=1)
    parser.add_argument("--use_feature_similarity", type=int, default=1)
    parser.add_argument("--use_embedding_similarity", type=int, default=1)

    parser.add_argument('--pg', type=str, default='reinforce')
    parser.add_argument('--ppo_epoch', type=int, default=5)

    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--schedule', type=int, default=0)

    args = parser.parse_args()
    logargs(args,tablename="config")
    args.pnhid = [int(n) for n in args.pnhid.split('+')]
    
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    return args

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)

    def forward(self, state):

        q1 = F.relu(self.l1(state))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

        # # Q2 architecture
        # self.l4 = nn.Linear(state_dim + action_dim, 256)
        # self.l5 = nn.Linear(256, 256)
        # self.l6 = nn.Linear(256, 1)

    def forward(self, state):
        q1 = F.relu(self.l1(state))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1
class SingleTrain(object):

    def __init__(self, args):
    
        self.globel_number = 1
        self.args = args
        self.datasets = self.args.datasets.split("+")
        self.budgets = [int(x) for x in self.args.budgets.split("+")]
        self.graphs, self.players, self.rshapers, self.accmeters = [], [], [], []
        for i, dataset in enumerate(self.datasets):
            g = GraphLoader(dataset, sparse=True,args=args, multigraphindex='graph' + str(i + 1))
            g.process()
            self.graphs.append(g)
            p = Player(g, args).cuda()
            self.players.append(p)
            self.rshapers.append(RewardShaper(args))
            self.accmeters.append(AverageMeter("accmeter",ave_step=100))
        self.env = Env(self.players,args)
        self.tau=0.005
        self.policy=switcher[args.policynet](self.args,self.env.statedim).cuda()

        self.opt = torch.optim.Adam(self.policy.parameters(), lr=self.args.rllr)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.opt, [1000, 3000], gamma=0.1, last_epoch=-1)
        self.accmeter = (AverageMeter("aveaccmeter",ave_step=100))


    def jointtrain(self, maxepisode):

        for episode in range(1,maxepisode):
            for playerid in range(len(self.datasets)):
                shapedrewards, logp_actions, p_actions = self.playOneEpisode(episode, playerid=playerid)
                if episode > 10:
                    loss = self.finishEpisode(shapedrewards, logp_actions, p_actions)
                else:
                    loss = None
                #if episode%self.args.logfreq == 0:
                    #logger.info("episode {}, playerid {}, loss {}".format(episode,playerid,loss))
                    #inspect_weight(self.policy)
                if self.args.save==1 and self.accmeter.should_save():
                    logger.warning("saving!")
                    torch.save(self.policy.state_dict(),"models/{}.pkl".format(self.args.policynet+self.args.savename))
            if (episode % 100 == 1):
                torch.save(self.policy.state_dict(),"models/{}.pkl".format(self.args.policynet+self.args.savename+'_'+str(episode)))                
            self.globel_number +=1

        
    def playOneEpisode(self, episode, playerid=0):

        self.playerid = playerid
        self.env.reset(playerid)
        rewards, logp_actions, p_actions = [], [], []
        self.states, self.actions, self.pools = [], [], []
        initialrewards=self.env.players[playerid].validation()
        rewards.append(initialrewards)
        self.entropy_reg = []
        self.action_index = np.zeros([self.args.batchsize, self.budgets[playerid]])
        for epoch in range(self.budgets[playerid]):
            state = self.env.getState(playerid)
            self.states.append(state)
            pool = self.env.players[playerid].getPool(reduce=False)
            self.pools.append(pool)
            logits = self.policy(state,self.graphs[playerid].normadj)
            #action indicates the next node we are going to annotate
            action,logp_action, p_action = self.selectActions(logits,pool)
            self.action_index[:, epoch] = action.detach().cpu().numpy()
            logp_actions.append(logp_action)
            p_actions.append(p_action)
            rewards.append(self.env.step(action,playerid))
            self.entropy_reg.append(-(self.valid_probs * torch.log(1e-6 + self.valid_probs)).sum(dim=1) / np.log(self.valid_probs.size(1)))
            ##
            # if episode % self.args.logfreq == 0:
            #     logger.debug("probs {}".format(logits[0].max()))
            #     logger.debug("action's state{} ".format(state[0,action[0],:]))
        #if episode % self.args.logfreq == 0:
            #print ('normalized entropy of the current policy')
            #print (self.entropy_reg.mean(dim=1) / np.log(logits.size(1)))
        
        self.env.players[playerid].trainRemain()
        logp_actions = torch.stack(logp_actions)
        p_actions = torch.stack(p_actions)
        self.entropy_reg = torch.stack(self.entropy_reg).cuda()
        finalrewards = self.env.players[playerid].validation(rerun=True)
        micfinal, _ = mean_std(finalrewards[0])
        self.accmeters[playerid].update(micfinal)
        self.accmeter.update(micfinal)
        if episode % self.args.logfreq == 0:
            logger.info("episode {},playerid{}. acc in validation {},aveacc {}".format(episode, playerid, micfinal, self.accmeters[playerid]()))
        shapedrewards = self.rshapers[playerid].reshape(rewards,finalrewards,logp_actions.detach().cpu().numpy())
        return shapedrewards,logp_actions, p_actions


    def finishEpisode(self,rewards,logp_actions, p_actions):

        rewards = torch.from_numpy(rewards).cuda().type(torch.float32)
        if (self.args.pg == 'reinforce'):
            #losses =torch.sum(-logp_actions*rewards,dim=0)
            #loss = torch.mean(losses) - self.args.entcoef * self.entropy_reg.sum(dim=0).mean()
            losses = logp_actions * rewards + self.args.entcoef * self.entropy_reg
            loss = -torch.mean(torch.sum(losses, dim=0))
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            if self.args.schedule:
                self.scheduler.step()
        elif (self.args.pg=='aac'):
            critic=Critic(self.env.statedim,self.actions[0].size(-1)).cuda()
            states_tensor=torch.stack(self.states)
            values=critic.forward(states_tensor)
            values=[torch.Tensor([values[i,0,0,0].item()]*(self.args.batchsize)).cuda() for i in range(self.budgets[self.playerid])]   
            values_tensor=torch.stack(values)
            values_loss=F.smooth_l1_loss(values_tensor,rewards)
            policy_loss=(-logp_actions*(rewards-values_tensor))
            losses=policy_loss+ values_loss+self.args.entcoef * self.entropy_reg
            loss=torch.mean(torch.sum(losses,dim=0))
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            if self.args.schedule:
                self.scheduler.step()
        elif (self.args.pg == 'ppo'):
            epsilon = 0.2
            p_old = p_actions.detach()
            for i in range(self.args.ppo_epoch):
                if (i != 0):
                    p_actions = [self.trackActionProb(self.states[i], self.pools[i], self.actions[i]) for i in range(len(self.states))]
                    p_actions = torch.stack(p_actions)
                ratio = p_actions / p_old
                surr1=ratio*rewards
                surr2=torch.clamp(ratio,1+epsilon,1-epsilon)*rewards
                losses = torch.min(surr1, surr2)
                loss = -torch.mean(losses)
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
        elif (self.args.pg=='sac'):
            #play one episode ensure that we are within budget:
            # last state is terminal state?
            self.critic_local1=Critic(self.env.statedim,self.actions[0].size(-1)).cuda()
            self.critic_local2=Critic(self.env.statedim,self.actions[0].size(-1)).cuda()
            self.critic_optimizer1=torch.optim.Adam(self.critic_local1.parameters(),self.args.lr)
            self.critic_optimizer2=torch.optim.Adam(self.critic_local2.parameters(),self.args.lr)
            #target network
            self.critic_target1=Critic(self.env.statedim,self.actions[0].size(-1)).cuda()
            self.critic_target2=Critic(self.env.statedim,self.actions[0].size(-1)).cuda()
            #equalize local and target network's weights
            self.update_weights(self.critic_local1,self.critic_target1)
            self.update_weights(self.critic_local2,self.critic_target2)
            self.actor_local=Actor(self.env.statedim,self.actions[0].size(-1)).cuda()
            self.actor_optimizer=torch.optim.Adam(self.actor_local.parameters(),self.args.lr)
            for i in range(self.args.ppo_epoch):
                next_states=[self.states[j+1] for j in range(len(self.states)-1)]
                next_states.append(self.states[len(self.states)-1])
                critic_loss1,critic_loss2=self.compute_critic_losses(self.states,torch.stack(next_states,dim=1),self.actions,rewards)
                critic_loss1.backward()
                critic_loss2.backward()
                self.critic_optimizer1.step()
                self.critic_optimizer2.step()

                actor_loss,log_probabilities=self.compute_actor_loss(self.states)
                actor_loss.backward()
                self.actor_optimizer.step()
                self.update_weights(self.critic_local1,self.critic_target1)
                self.update_weights(self.critic_local2,self.critic_target2)
            return actor_loss.item(),critic_loss1.item(),critic_loss2.item()
        return loss.item()

    def compute_critic_losses(self,states,next_states,actions,rewards):
        logits_list = [self.policy(state,self.graphs[self.playerid].normadj) for state in states]
        next_actions=[]
        p_probs=[]
        log_probs=[]
        for i in range(len(logits_list)):
          a,b,c=self.selectActions(logits_list[i],self.pools[i])
          next_actions.append(a)
          log_probs.append(b)
          p_probs.append(c)
        
        next_actions_tensor=torch.stack(next_actions,dim=1)
        p_probs_tensor=torch.stack(p_probs,dim=1)
        log_probs_tensor=torch.stack(log_probs,dim=1)
        with torch.no_grad():
            Q1_target=self.critic_target1.forward(next_states).squeeze(-1).mean(-1)
            Q2_target=self.critic_target2.forward(next_states).squeeze(-1).mean(-1)
            min_Q=p_probs_tensor*(torch.min(Q1_target,Q2_target)-log_probs_tensor)
           
            not_done = torch.ones_like(Q1_target).cuda()
            not_done[self.args.batchsize-1,:]=torch.zeros(Q1_target[0].size()).cuda()
            next_Qs=rewards.T+not_done*0.99*min_Q
        states_tensor=torch.stack(states,1)
        Q1s=self.critic_local1.forward(states_tensor).squeeze(-1).mean(-1)
        Q2s=self.critic_local2.forward(states_tensor).squeeze(-1).mean(-1)
        Q1_loss=F.mse_loss(Q1s,next_Qs)
        Q2_loss=F.mse_loss(Q2s,next_Qs)
        return Q1_loss,Q2_loss
    def compute_actor_loss(self,states):
        logits_list = [self.policy(state,self.graphs[self.playerid].normadj) for state in states]
        log_probs=[]
        probs=[]
        for i in range(len(logits_list)):
          a,b,c=self.selectActions(logits_list[i],self.pools[i])
          log_probs.append(b)
          probs.append(c)
        log_probs_tensor=torch.stack(log_probs,1)
        probs_tensor=torch.stack(probs,1)
        states_tensor=torch.stack(states,1)
        Q1s=self.critic_local1(states_tensor).squeeze(-1).mean(-1)
        Q2s=self.critic_local2(states_tensor).squeeze(-1).mean(-1)
        minQ=torch.min(Q1s,Q2s)
        policy_loss=torch.sum(log_probs_tensor-minQ,dim=1).mean()
        probabilities=torch.sum(log_probs_tensor*probs_tensor,1)
        return policy_loss,probabilities
    
    def trackActionProb(self, state, pool, action):
        logits = self.policy(state, self.graphs[self.playerid].normadj)
        valid_logits = logits[pool].reshape(self.args.batchsize,-1)
        max_logits = torch.max(valid_logits,dim=1,keepdim=True)[0].detach()
        valid_logits = valid_logits - max_logits
        valid_probs = F.softmax(valid_logits, dim=1)
        prob = valid_probs[list(range(self.args.batchsize)), action]
        return prob


    def selectActions(self,logits,pool):
        valid_logits = logits[pool].reshape(self.args.batchsize,-1)
        max_logits = torch.max(valid_logits,dim=1,keepdim=True)[0].detach()
        valid_logits = valid_logits - max_logits
        valid_probs = F.softmax(valid_logits, dim=1)
        self.valid_probs = valid_probs
        pool = pool[1].reshape(self.args.batchsize,-1)
        assert pool.size()==valid_probs.size()

        m = Categorical(valid_probs)
        action_inpool = m.sample()
        self.actions.append(action_inpool)
        logprob = m.log_prob(action_inpool)
        prob = valid_probs[list(range(self.args.batchsize)), action_inpool]
        action = pool[[x for x in range(self.args.batchsize)],action_inpool]
    
        return action, logprob, prob

    def update_weights(self,local_network,target_network):
        for local_param, target_param in zip(local_network.parameters(),target_network.parameters()):
            target_param.data.copy_(self.tau*local_param.data+(1-self.tau)*target_param.data)

if __name__ == "__main__":

    args = parse_args()
    singletrain = SingleTrain(args)
    singletrain.jointtrain(args.maxepisode)   