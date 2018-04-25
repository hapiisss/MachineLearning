import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

rw = pd.read_csv('data.txt', delimiter = '\t', header = None)
rw = np.array(rw).tolist()

def build_R(rw):
    xlen = len(rw)
    ylen = len(rw[0])

    d = pd.DataFrame(columns = ['up', 'right', 'down', 'left', 'this'])
    idx = 0 
    
    for i in range(ylen):
        for j in range(xlen):
            move = [float('-inf'),float('-inf'),float('-inf'),float('-inf'), rw[i][j]]
            if i>0: #up
                move[0] = rw[i-1][j]
            if i<ylen-1: #down
                move[2] = rw[i+1][j]
            if j>0: #left
                move[3] = rw[i][j-1]
            if j<xlen-1: #right
                move[1] = rw[i][j+1]
            d.loc[idx] = move
            idx+=1
    return d
    

R = build_R(rw)
R = np.array(R)
Q = np.zeros((100,5))

def build_trans(rw):
    xlen = len(rw)
    ylen = len(rw[0])
    
    d = {'up':[] ,'right':[], 'down':[], 'left':[]}
    d = pd.DataFrame(columns = ['up', 'right', 'down', 'left', 'none'])
    idx = 0 
    
    for i in range(ylen):
        for j in range(xlen):
            trans = [-1,-1,-1,-1, idx]
            if i>0: #up
                trans[0] = idx-xlen
            if i<ylen-1: #down
                trans[2] = idx+xlen
            if j>0: #left
                trans[3] = idx-1
            if j<xlen-1: #right
                trans[1] = idx+1
            d.loc[idx] = trans
            idx+=1
    return d


trans = build_trans(rw)
trans = np.array(trans)

def build_va(trans):
    va = []
    for i in range(len(trans)):
        current_va = []
        if (trans[i][0] != -1): #up
            current_va.append(0)
        if (trans[i][1] != -1): #right
            current_va.append(1)
        if (trans[i][2] != -1): #down
            current_va.append(2)
        if (trans[i][3] != -1): #left
            current_va.append(3)
        current_va.append(4)
        va.append(current_va)
    return va

	va = build_va(trans)
va = np.array(va)

# # Main Program



gamma = 0.8
episodes = 10000
reward_list = []
for i in range(episodes):
    start_state = 90
    goal_state = 9
    current_state = start_state
    current_reward = R[current_state][4]
    while current_state != goal_state:
        action = random.choice(va[current_state])
        next_state = trans[current_state][action]
        future_rewards = []
        for action_nxt in va[next_state]:
            future_rewards.append(Q[next_state][action_nxt])
            
        #update Q
#         print('CS : {}\nAct : {}'.format(current_state, action))
        qstate = R[current_state][action] + gamma*max(future_rewards)
        Q[current_state][action] = qstate
#         print(Q)
        current_state = next_state
        current_reward += R[current_state][4]
        if(current_state == goal_state):
            print('Episode {}, score : {}'.format(i, current_reward))
            reward_list.append(current_reward)





Q_df = pd.DataFrame(columns = ['up', 'right', 'down', 'left', 'none'])

# In[19]:


for i in range(len(Q)):
    Q_df.loc[i] = Q[i]



print(max(reward_list))

