from openpyxl import load_workbook
import random
import numpy as np
import pandas as pd

filepath="D:/Shanghai Maritime/LUSC/clinic/time.txt"

file=open(filepath,"r")
file.readline()
line=file.readline()
alivesum=[]
deadsum=[]
deadline=720
other=0
while line:
    if int(line.split()[1])>deadline:
        alivesum.append(line.split()[0])   #0
    elif int(line.split()[1])<deadline and int(line.split()[2])==1 and int(line.split()[1])!=0:
        deadsum.append(line.split()[0])    #1
    else:
        other+=1
    line=file.readline()
file.close()

print(len(alivesum))
print(len(deadsum))
print(alivesum)
print(other)

persondict={}
for i in alivesum:
    persondict[i]=0
for i in deadsum:
    persondict[i]=1
print(persondict)
print(len(persondict.keys()))
# np.save("../data/person_dict.npy",persondict)
# np.save("../data/alive_person.npy",np.array(alivesum))
# np.save("../data/dead_person.npy",np.array(deadsum))




