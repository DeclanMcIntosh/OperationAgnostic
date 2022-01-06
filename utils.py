
### General
def f2int(float):
    return int(float*128 + 128)

def int2f(int):
    return (int - 128)/128.

def IoU(a, b):
    return len(list(set(a) & set(b)))/len(list(set(a) | set(b)))
    
#### Swing up
def convertSwingUpActionState(obs):
    output = []
    for x in range(obs.shape[0]):
        output.append(f2int(obs[x]))
    return output

#### Cart-Pole V0
def convertCartPoleActionState(obs):
    output= [f2int(obs[0]/2.4),f2int(obs[1]/10),f2int(obs[2]/0.20943951),f2int(obs[3]/10)]
    return output

def CartPole_Action(obs, agent):
    uintObs = convertCartPoleActionState(obs)
    if agent.forward(uintObs)[0] > 128: action = 1
    else: action = 0
    return action

def CartPole_Action_Binary(obs, agent):
    fullObs = []
    uintObs = convertCartPoleActionState(obs)
    for value in uintObs:
        temp = value
        for x in list(range(8)):
            if temp >= 2**(7-x):
                fullObs.append(1)
                temp = temp - 2**(7-x)
            else:
                fullObs.append(0)

    return agent.forward(fullObs)[0]

#### dispatcher for config files to be able to find functions
dispatcher = {  "CartPole_Action": CartPole_Action,
                "CartPole_Action_Binary": CartPole_Action_Binary,
                "convertSwingUpActionState": convertSwingUpActionState}