import gym 
import random
import time

env=gym.make('GridWorld-v0')

class Learn:
    def __init__(self,grid_mdp):
        #初始化状态值函数
        self.v=dict()
        for state in grid_mdp.states:
            self.v[state]=0
        #初始化策略
        self.pi=dict()
        #random.choice(seq):返回列表、元组、字符串的随机项
        self.pi[1]=random.choice(['e','s'])
        self.pi[2]=random.choice(['e','w'])
        self.pi[3]=random.choice(['w','s','e'])
        self.pi[4]=random.choice(['e','w'])
        self.pi[5]=random.choice(['w','s'])
    #策略迭代函数
    def policy_iterate(self,grid_mdp):
        for i in range(100):
            #策略评估和策略改善交替进行
            self.policy_evaluate(grid_mdp)
            self.policy_improve(grid_mdp)
    #策略评估函数
    def policy_evaluate(self,grid_mdp):
        #迭代法求解线性方程组
        for i in range(1000):
            delta=0.0
            #遍历状态空间
            for state in grid_mdp.states:
                if state in grid_mdp.terminate_states:
                    continue
                action=self.pi[state]
                t,s,r=grid_mdp.transform(state,action)
                new_v=r+grid_mdp.gamma*self.v[s]
                delta+=abs(new_v-self.v[state])
                self.v[state]=new_v
            if delta < 1e-6:
                break
    #策略改善函数:寻找动作空间中最优动作
    def policy_improve(self,grid_mdp):
        #在每个状态下采用贪婪策略
        for state in grid_mdp.states:
            if state in grid_mdp.terminate_states:
                continue
            #假设第一个动作为最优动作
            action=grid_mdp.actions[0]
            t,s,r=grid_mdp.transform(state,action)
            #最优状态值函数=最优状态动作值函数
            v=r+grid_mdp.gamma*self.v[s]
            #遍历动作空间与最优动作进行比较，从而找到最优动作
            for action in grid_mdp.actions:
                t,s,r=grid_mdp.transform(state,action)
                if v < r+grid_mdp.gamma*self.v[s]:
                    a=action
                    v=r+grid_mdp.gamma*self.v[s]
            self.pi[state]=a
    def action(self,state):
        return self.pi[state]

gm=env.env
state=env.reset()
#实例化对象
learn=Learn(gm)
#策略迭代
learn.policy_evaluate(gm)
total_reward=0
for i in range(100):
    env.render()
    action=learn.action(state)
    state,reward,done,_=env.step(action)
    total_reward+=reward 
    time.sleep(5)
    if done:
        break



