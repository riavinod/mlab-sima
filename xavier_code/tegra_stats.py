import pandas as pd
import numpy as np
import sys

def tegra_stats(log_name):
    
    columns = ['RAM','GPU Util', 'GPU', 'Power']
    df = pd.DataFrame(columns=columns)
 
    count = 0
    with open(log_name, 'r') as f:
        for line in f:
            row = get_series(line)
            if type(row)!=int:
                df = df.append(row, ignore_index=True)
            count += 1
    
#     print(log_name)
#     #print("Total number of lines is:", count)
#     print('')
    print(df)
    print_stats(df)
    return df
    
def get_series(line):
    line = line.split()
    row = []
    dict = {}
    for i in range (0, len(line)):
    
        if line[i]=="RAM":
            x = line[i+1].split('/')[0]
            #print(line[i], line[i+1])
            row.append(int(x))
            dict["RAM"] = int(x)

        elif line[i]=="VDDRQ":
            x = line[i+1].split('/')[0]
            #print(line[i], line[i+1])
            row.append(int(x)/1000)
            dict["VDDRQ"] = int(x)

        elif line[i]=="GR3D_FREQ":
            x = line[i+1][0][:1]
            #print(line[i], line[i+1])
            row.append(int(x))
            dict["GR3D_FREQ"] = int(x)

        elif line[i]=="GPU":
            x = line[i+1].split('/')[0]
            #print(line[i], line[i+1])
            row.append(int(x))
            dict["GPU"] = int(x)
    
    #print(" ")
    
    
    if dict["GR3D_FREQ"]!=0:
        #print(dict)
        return dict
    return -1

def print_stats(df):
    means = df.mean(axis=0)
    maxes = df.max(axis=0)
    print("Mean RAM: ", means[0])
    print("Mean GPU Util: ", means[4])
    print("Mean Power : ", means[5])
    print("Max Power: ", maxes[5])    


for arg in sys.argv[1:]:
    print(arg)
    tegra_stats(arg)
    print('')
