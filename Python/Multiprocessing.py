#Manager.list/dict + Process
import pandas as pd
from multiprocessing import Process, Manager
mydata = pd.read_csv('data/data45265/train_data.csv', header = None, encoding='utf8')
titles = mydata.loc[0,0].split('|')
length = len(mydata)
scale = list(range(1,length+1,int(length/4)))
scale[-1] = length
def appendmydata(mydata, results):
    temp = []
    for idx, row in mydata.iterrows():
        temp.append(row[0].split('|'))
    results.append(temp)
mgr = Manager()
results = mgr.list()
processPool = []
for i in range(len(scale) - 1):
    temp_data = mydata.iloc[scale[i]:scale[i + 1]]
    t = Process(target = appendmydata, args = (temp_data, results))
    processPool.append(t)
    t.start()
for process in processPool:
    process.join()


#Pool + apply_async (with return)
import pandas as pd
from multiprocessing import Pool
mydata = pd.read_csv('data/data45265/train_data.csv', header = None, encoding='utf8')
titles = mydata.loc[0,0].split('|')
length = len(mydata)
scale = list(range(1,length+1,int(length/12)))
scale[-1] = length
def appendmydata(mydata):
    temp = []
    for idx, row in mydata.iterrows():
        temp.append(row[0].split('|'))
    return temp
p = Pool(12)
results = []
for i in range(len(scale) - 1):
    temp_data = mydata.iloc[scale[i]:scale[i + 1]]
    res = p.apply_async(appendmydata, args = (temp_data, ))
    results.append(res)
p.close()
p.join()