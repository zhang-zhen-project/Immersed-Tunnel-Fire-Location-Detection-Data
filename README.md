# zhangzhen
Immersed-Tunnel-Fire-Detection-Data-Set
# CFD-Data: 
This is a project that encompasses 6 ignition source locations and 32 different smoke exhaust configurations, resulting in a total of 192 FDS computational outcomes. Each individual result is stored in a CSV file, with the file name following a specific pattern: for instance, the filename "100M32" signifies an ignition source located at Y=100 meters, "M" indicating the middle of the ignition lane, "U" denoting one side of the lane, and "32" representing the 32nd smoke exhaust configuration as described in the paper.
# ReadDataSet.py:
This is a program that reads CFD-Data and automatically generates a fire source recognition task dataset.
# FireLocartionDetection.py:
This is a program designed to identify the location of fire sources and smoke exhaust configurations. It incorporates self-built structures and hyperparameters for BPNN, CNN, LSTM, BiLSTM, CNN-LSTM, and CNN-BiLSTM models, which were determined through grid search to achieve optimal performance. The evaluation metrics include accuracy, precision, recall, and F1 score.

