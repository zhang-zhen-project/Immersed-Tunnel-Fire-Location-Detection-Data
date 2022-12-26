import numpy as np
import pandas as pd
import os

# Define Root catalogue of datasets
FatherPath = ("D:\Code\FIRE\FIRE")
DataPath = (FatherPath + "/DATA")


# Define fire location
def MapLoc(key):
    global ValueKey
    ValueKey = dict([('70M', 1), ('70U', 2), ('100M', 3), ('100U', 4), ('130M', 5), ('130U', 6)])
    return ValueKey.get(key)


# Get feature and label
def ScanerFile(url, InitPreDf, InitPreLabelLoc, InitPreLabelDam, total, slide, list_c, t):
    file = os.listdir(url)
    # 初始化一个DataFrame空表
    for f in file:
        if (f.find("70") != -1 or f.find("100") != -1 or f.find("130") != -1) and (f.find("M") != -1):
            InitPreDf = pd.read_csv(DataPath + "/" + f, usecols=list_c
                                    # range(1, 201)
                                    , skiprows=2, nrows=1,
                                    engine='python',
                                    header=None)
        InitPreDf = InitPreDf.drop(index=InitPreDf.index)
        break
    for f in file:
        if f.find("70") != -1 or f.find("100") != -1 or f.find("130") != -1:
            print("Reading: " + f, end=' ')
            # for i in range(TimeSteps - Slide):
            df = pd.read_csv(DataPath + "/" + f, usecols=list_c
                             # range(1, 201)
                             , skiprows=3, nrows=total,
                             engine='python',
                             dtype=object,
                             header=None)

            for i in range(total - slide):
                InitPreDf = pd.concat([InitPreDf, df.loc[i:i + slide - 1, :]], axis=0)
                InitPreLabelLoc.append(MapLoc(str(f[:-11])))
                InitPreLabelDam.append(int(f[-11:-9]))
            # df["label"] = PreLabel
            # print('Row :' + str(InitPreDf.shape[0]) + "\nClo:" + str(InitPreDf.shape[1]))
            print("Down\n")

            # InitPreLabel = np.array(InitPreLabel)
    InitArrayDf = np.array(InitPreDf)
    # InitArrayDf = InitArrayDf.reshape(len(InitPreLabel), Slide, 200)
    InitArrayLabelLoc = np.array(InitPreLabelLoc)
    InitArrayLabelDam = np.array(InitPreLabelDam)
    np.save('DATASET/Temperature/InitArrayDf' + t + '.npy', InitArrayDf)
    np.save('DATASET/Temperature/InitArrayLabelLoc' + t + '.npy', InitArrayLabelLoc)
    np.save('DATASET/Temperature/InitArrayLabelDam' + t + '.npy', InitArrayLabelDam)


# Define init dataSets
InitPreDf = pd.DataFrame()
InitPreLabelLoc = []
InitPreLabelDam = []
a = list(range(1, 201))
b = list(range(406, 606))
c = a + b
t = 8

ScanerFile(DataPath, InitPreDf, InitPreLabelLoc, InitPreLabelDam, int(2 * t), 4, c, str(t))
