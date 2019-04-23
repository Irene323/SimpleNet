import random

data=[]
with open('../resources/merge_200000.txt', 'r') as f:
    for l in f:
        data.append(l)
o = []
while len(data) > 0:
    i = int(random.uniform(0, len(data)))
    o.append(data[i])
    data.pop(i)
with open('../resources/merge_200000_train.txt', 'w') as wtrain:
    with open('../resources/merge_200000_test.txt', 'w') as wtest:
        for j in range(0, len(o)):
            if j<43515:
                wtrain.write(o[j])
            else:
                wtest.write(o[j])

    # for d in o:
    #     fw.write(d)