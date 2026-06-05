import nnetga
from time import perf_counter,sleep


agentNum = 50
nnetga.add_pop(1,0.7,0.02,0.3)
nnetga.add_agent(0,agentNum)
nnetga.add_net(0,2,1,3,1)

#chromo = [6,-6,-3,-6,5,-3,10,10,-5]
#chromo = [0.6,-0.6,-0.3,-0.6,0.5,-0.3,0.10,0.10,-0.5]
#nnetga.insert_chromo(0,10,chromo)
#nnetga.insert_chromo(0,11,chromo)
#nnetga.insert_chromo(0,12,chromo)
#nnetga.insert_chromo(0,13,chromo)
#nnetga.insert_chromo(0,14,chromo)
#nnetga.insert_chromo(0,15,chromo)
#nnetga.info_net(0,13,1)

cases = [
    ([0, 1], 1),
    ([0, 0], 0),
    ([1, 1], 0),
    ([1, 0], 1),
]


def batch(inputs):
    return [[list(inputs) for _ in range(agentNum)]]


def score_outputs(outputs):
    scores = [0.0] * agentNum
    for output, target in outputs:
        for iAgent, value in enumerate(output[0]):
            error = target - value[0]
            scores[iAgent] += (1.0 - error * error) * 100.0
    return scores


bestScore = -1.0
bestAgent = 0
bestChromo = None

for i in range(1000):
    outputs = [(nnetga.update(batch(inputs)), target) for inputs, target in cases]
    Score = score_outputs(outputs)
    
    #for i in range(agentNum):
        #print(a[0][i][0])
        #print(b[0][i][0])
        #print(c[0][i][0])
        #print(d[0][i][0])

    '''
    for iAgent in range(agentNum):
        if a[0][iAgent][0] > 0.5:
            Score[iAgent] += 1
        if b[0][iAgent][0] < 0.5:
            Score[iAgent] += 1
        if c[0][iAgent][0] < 0.5:
            Score[iAgent] += 1
        if d[0][iAgent][0] > 0.5:
            Score[iAgent] += 1
        if Score[iAgent] == 4:
            #print(iAgent,"have 4 at gen ",i)
            Score[iAgent] += 1
    '''   
    hiScore = max(Score)
    Average = sum(Score) / agentNum
    WinnerAgent = Score.index(hiScore)
    if hiScore > bestScore:
        bestScore = hiScore
        bestAgent = WinnerAgent
        bestChromo = nnetga.get_chromo(0, WinnerAgent)
    #print ("Agent:",Score.index(hiScore)," winner with:", hiScore)
    #print ("Average score:",Average)
    #nnetga.info_net(0,Score.index(hiScore),1)
    #print("Number 13 score:",Score[13])
    #print("----")
    nnetga.next_gen(0,Score,EliteNum=1,WorstNum=1)
    print(Average,end="")
    back = "\b" * len(str(Average))
    print(back,end="")
    
nnetga.insert_chromo(0,0,bestChromo)
trainedOutputs = [nnetga.update(batch(inputs))[0][0][0] for inputs, target in cases]

print ("Agent:",bestAgent," winner with:", bestScore)
print ("Average score:",Average)
nnetga.info_net(0,0,1)
for (inputs, target), output in zip(cases, trainedOutputs):
    print("input",inputs,"-> output",output,"target",target)
print("----")

'''
agentNum = 4000
stime = clock()
nnetga.add_pop()
print ("Pop added in: ",clock() - stime,"sec")
stime = clock()
nnetga.add_agent(0,agentNum)
print ("Agent added in: ",clock() - stime,"sec")
stime = clock()
nnetga.add_net(0,2,1,2,1)
print ("net added in: ",clock() - stime,"sec")
nnetga.info_net(0,0,2)

c = [6,-6,-3,-6,5,-3,10,10,-5]
stime = clock()
nnetga.insert_chromo(0,c)
print ("Insert Chromo in: ",clock() - stime,"sec")

a = [0,1]
a = [a] * agentNum
a = [a]
#print(a)
stime = clock()
b = nnetga.update(a)
#print("0 1:",b)
print ("net updated in: ",clock() - stime,"sec")

d = [0] * agentNum
for i in range(agentNum):
    d[i] = i
stime = clock()    
nnetga.next_gen(0,d)
print ("new generation in: ",clock() - stime,"sec")
nnetga.info_net(0,0,2)

#for i in range(agentNum):
    #nnetga.info_net(0,i,1)
'''
