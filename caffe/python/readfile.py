f=open('result/cluster-100.txt','r')
count = 0
ints = []
i = 2
data = f.readlines()
while i <= len(data):
    # print data[i].strip()
    ints.append(int(data[i].strip()))
    i = i + 3
for i in range(len(ints)):
    count = count + ints[i]
count = count / 790
print('counts:')
print count


