import numpy as np
import pandas as pd
import os

# Define Root catalogue of datasets
FatherPath = ("D:\code\FIRE")
DataPath = (FatherPath + "/DATA")


# Define fire location
def MapLoc(key):
    global ValueKey
    ValueKey = dict([('70M', 1), ('70U', 2), ('100M', 3), ('100U', 4), ('130M', 5), ('130U', 6)])
    return ValueKey.get(key)


# Get feature and label
def ScanerFile(url, InitPreDf, InitPreLabelLoc, InitPreLabelDam, total, slide, list_c, t):
    file = os.listdir(url)
    ArrayDFList = []
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
                a = np.array(df.loc[i:i + slide - 1, :])
                ArrayDFList.append(a)

                # InitPreDf = pd.concat([InitPreDf, df.loc[i:i + slide - 1, :]], axis=0)
                InitPreLabelLoc.append(MapLoc(str(f[:-11])))
                InitPreLabelDam.append(int(f[-11:-9]))
            # df["label"] = PreLabel
            # print('Row :' + str(InitPreDf.shape[0]) + "\nClo:" + str(InitPreDf.shape[1]))
            print("Down\n")

            # InitPreLabel = np.array(InitPreLabel)
    for num in ArrayDFList:
        if num.shape[0] != slide:
            print('FALSE')
    InitArrayDf = np.stack(ArrayDFList)
    # InitArrayDf = np.array(InitPreDf)
    InitArrayDf = InitArrayDf.reshape(-1, len(list_c))
    InitArrayLabelLoc = np.array(InitPreLabelLoc)
    InitArrayLabelDam = np.array(InitPreLabelDam)
    np.save('./DATASET/InitArrayDf' + t + '-' + str(slide) + str(len(list_c)) + '.npy', InitArrayDf)
    np.save('./DATASET/InitArrayLabelLoc' + t + '-' + str(slide) + str(len(list_c)) + '.npy',
            InitArrayLabelLoc)
    np.save('./DATASET/InitArrayLabelDam' + t + '-' + str(slide) + str(len(list_c)) + '.npy',
            InitArrayLabelDam)

# Define init dataSets
InitPreDf = pd.DataFrame()
InitPreLabelLoc = []
InitPreLabelDam = []

# select sensor numbers
soot_sensors = list([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
temp_sensors = list([213, 220, 230, 240, 250, 260, 270, 278, 280, 290])
sensors = soot_sensors + temp_sensors
# select duration of fire
time = 300
# set time windows
slide = 30
# generate dataset
ScanerFile(DataPath, InitPreDf, InitPreLabelLoc, InitPreLabelDam, int(2 * time), slide, sensors, str(time))
