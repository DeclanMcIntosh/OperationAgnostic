from math import floor

### General
def f2int(float):
    return int(float*128 + 127)

def int2f(int):
    return (int - 128)/128.

def IoU(a, b):
    return len(list(set(a) & set(b)))/len(list(set(a) | set(b)))
    
def binatodeci(binary):
    return sum(val*(2**idx) for idx, val in enumerate(reversed(binary)))

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
    uintObs = convertCartPoleActionState(obs)#[0:3]
    for value in uintObs:
        temp = value
        for x in list(range(8)):
            if temp >= 2**(7-x):
                fullObs.append(1)
                temp = temp - 2**(7-x)
            else:
                fullObs.append(0)
    # Hacky to force binary methods out of loca minima
    if agent.forward(fullObs)[0] != agent.forward(fullObs)[1]:
        return 1 
    else: 
        return 0

def CartPole_Action_Binary_Binned(obs, agent):
    bins = 16 
    fullObs = []
    uintObs = convertCartPoleActionState(obs)
    for value in uintObs:
        x = floor(value/(bins))
        for n in range(bins):
            if x!=n: fullObs.append(0)
            else: fullObs.append(1)
    
    if agent.forward(fullObs)[0] != agent.forward(fullObs)[1]:
        return 1 
    else: 
        return 0


#### Bipedal-Walker
def convertWalkerActionState(obs):
    output= [   f2int(obs[0]/4.25),f2int(obs[1]/0.31),f2int(obs[2]/1.),f2int(obs[3]/1.),f2int(obs[4]/3.),
                f2int(obs[5]/3.),f2int(obs[6]/3.),f2int(obs[7]/9.),f2int(obs[8]/1.),f2int(obs[9]/3.),
                f2int(obs[10]/3.),f2int(obs[11]/3.),f2int(obs[12]/15.),f2int(obs[13]/1.),f2int(obs[14]/1.),
                f2int(obs[15]/1.),f2int(obs[16]/1.),f2int(obs[17]/1.),f2int(obs[18]/1.),f2int(obs[19]/1.),
                f2int(obs[20]/1,),f2int(obs[21]/1.),f2int(obs[22]/1.),f2int(obs[23]/1.),]

    if max(output) > 255: print("BAD VALUE TOO BIG -> Index: " + str(output.index(max(output))) + " Value: " + str(obs[output.index(max(output))]))
    if min(output) < 0: print("BAD VALUE TOO SMALL -> Index: " + str(output.index(min(output))) + " Value: " + str(obs[output.index(min(output))]))
    return output

def Walker_Action(obs, agent):
    uintObs = convertWalkerActionState(obs)
    action = agent.forward(uintObs)
    output = []
    for value in action:
        output.append(int2f(value))

    return output

def Walker_Action_Binary(obs, agent):
    uintObs = convertWalkerActionState(obs)
    fullObs = []
    for value in uintObs:
        temp = value
        for x in list(range(8)):
            if temp >= 2**(7-x):
                fullObs.append(1)
                temp = temp - 2**(7-x)
            else:
                fullObs.append(0)
    action = agent.forward(fullObs)

    output = []

    for x in range(4):
        output.append((binatodeci(action[8*x:8*(x+1)])-128)/128)

    return output

#### LunarLander-v2
def convertLunarLanderActionState(obs):
    return list(map(f2int, obs))

def LunarLander_Action(obs, agent):
    uintObs = convertLunarLanderActionState(obs)

    modelActions = agent.forward(uintObs)

    maxValue = max(modelActions)

    return modelActions.index(maxValue)




#### dispatcher for config files to be able to find functions
dispatcher = {  "CartPole_Action": CartPole_Action,
                "CartPole_Action_Binary": CartPole_Action_Binary,
                "convertSwingUpActionState": convertSwingUpActionState,
                "Walker_Action": Walker_Action,
                "Walker_Action_Binary": Walker_Action_Binary,
                "LunarLander_Action": LunarLander_Action,
                "convertSwingUpActionState": convertSwingUpActionState,
                "CartPole_Action_Binary_Binned":CartPole_Action_Binary_Binned}
