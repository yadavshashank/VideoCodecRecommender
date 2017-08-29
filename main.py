from __future__ import division
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

def retrieval(a,b,c,d):
    File =open('transcoding3.csv','r')
    allLines=File.readlines()
    File.close()
    
    allLines=[i.split() for i in allLines]
   
    dataList=[]
    for i in range(0,len(allLines)):
        if int(allLines[i][17])==a and int(allLines[i][18])==b and int(allLines[i][15])==c and float(allLines[i][16])==d :
            newI=','.join(allLines[i])
            dataList.append(newI)
    return dataList


def store(x):
    File2=open('data300.txt','w')
    for i in x:
        File2.write(i)
        File2.write("\n")
    File2.close()

def calc(length):
    path = os.getcwd() + '/data300.txt'
    source = pd.read_csv(path,names=['Duration','Codec', 'Width', 'Height','Bitrate','Framerate','i','p','b','Frame','i_size','p_size','b_size','Size','Output Codec','ob','of','ow','oh','umem','utime'])
        
    #scatter plot
    source.plot(kind='scatter', x='utime', y='Bitrate', figsize=(12,8)) 
    
    # We drop the columns which either contain zero values or have repeating values
    source.drop(source.columns[[8,12,15,16,17,18]],axis=1, inplace=True)
    
    # It does normalization of the entire dataset by subtracting from mean and dividing by standard variance
    for k in range(0,12):
        source.iloc[:,k] = (source.iloc[:,k] - (source.iloc[:,k]).mean()) / (source.iloc[:,k]).std()
    
    # Splitting the data set in the ration 80% (training) : 20% (testing)
    splitting_length = int(0.2*length)
    
    
    # Assigning 80% to train
    train = source.iloc[0:splitting_length,:]
    # Assigning 20% to test
    test = source.iloc[splitting_length:,:]
    
    # add ones column
    train.insert(0, 'Ones', 1)
   
    #Setting alpha or step-size
    alpha = 0.007
    
    #Setting no. of iterations
    iters = 5000
    
    # set X (training data) and y (target variable)
    cols = train.shape[1]
    X2 = train.iloc[:,0:cols-2]  
    y2_time = train.iloc[:,cols-1:cols]
    
    
    # convert to matrices and initialize theta
    X2 = np.matrix(X2.values)  
    y2_time = np.matrix(y2_time.values)  
    theta2 = np.matrix(np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0]))
    
    # perform linear regression on the data set
    g2, cost2 = gradientDescent(X2, y2_time, theta2, alpha, iters)
    
    # get the cost (error) of the model
    cost_time = computeCost(X2, y2_time, g2)
    print "Error :  ", cost_time

    #Plotting a curve for Cost vs Iterations    
    fig,ax = plt.subplots(figsize=(20,8))
    ax.plot(np.arange(iters), cost2, 'r')  
    ax.set_xlabel('Iterations')  
    ax.set_ylabel('Cost')
   
    # Generating a testlist for all the test data
    testList=[]
    for index, row in test.iterrows():
        testList.append(row)
    
      
    predictedCost=g2[0,0]
   
    # Recommending the codec for this test data

    small_time=100000
    small_codec=0
    
    for j in range(1,5):
        
        for i in range(1,13):
            # Multiplying weights with the feature value
            predictedCost+=(g2[0,i]*testList[25][i-1])
        
        predictedCost+=(g2[0,13]*j)
        
        # Determining the smallest time for each codec        
        if(small_time>predictedCost and predictedCost>0):
            small_time = predictedCost
            small_codec=j
        
        

    # Printing the smallest time taken    
    print "\nLeast amount of time taken to convert the video  = ",small_time
    
    #Printing the best codec for conversion
    if(small_codec==1):
        print("\nConvert to output codec MPEG4")
    elif(small_codec==2):
        print("\nConvert to output codec H264")
    elif(small_codec==3):
        print("\nConvert to output codec VL8")
    elif(small_codec==4):
        print("\nConvert to output codec FLV")


def gradientDescent(X, y, theta, alpha, iters):
    
    #Constructing a temporary matrix of zeros of theta
    temp = np.matrix(np.zeros(theta.shape))
    
    #Getting the no. of parameters in theta matrix, ravel is used to flatten a numpy array
    parameters = int(theta.ravel().shape[1])
    cost = np.zeros(iters)

    for i in range(iters):
        error = (X * theta.T) - y

        for j in range(parameters):
            term = np.multiply(error, X[:,j])
            temp[0,j] = theta[0,j] - ((alpha / len(X)) * np.sum(term))

        theta = temp
        
        # Assigning cost in the list
        cost[i] = computeCost(X, y, theta)

    return theta, cost


def computeCost(X, y, theta):
    inner = np.power(((X * theta.T) - y), 2)
    return np.sum(inner) / (2 * len(X))

print "\n Select video resolution:\n"
print "1-176 x 144\n"
print "2-480 x 360\n"
print "3-1280 x 720\n"
print "4-320 x 240\n"
print "5-640 x 480\n"
print "6-1920 x 1080\n"
print "0-Exit\n"
option=input("Enter your option : ")

if option ==1:
    width=176
    height=144
    bitrate= 56000
    framerate= 12.00
    myFile=retrieval(width,height,bitrate,framerate)
    
    store(myFile)
    calc(len(myFile))
    
elif option ==2:
    width=480
    height=360
    bitrate= 109000
    framerate= 15.00
    myFile=retrieval(width,height,bitrate,framerate)
    
    store(myFile)
    calc(len(myFile))
    
elif option ==3:
    width=1280
    height=720
    bitrate=242000
    framerate= 24.00
    myFile=retrieval(width,height,bitrate,framerate)
    
    store(myFile)
    calc(len(myFile))
    
elif option ==4:
    width=320
    height=240
    bitrate= 539000
    framerate= 25.00
    myFile=retrieval(width,height,bitrate,framerate)
    
    store(myFile)
    
    calc(len(myFile))
    
elif option ==5:
    width=640
    height=480
    bitrate= 820000
    framerate= 29.97
    myFile=retrieval(width,height,bitrate,framerate)
    
    store(myFile)
    calc(len(myFile))
    
elif option==6:
    width=1920
    height=1080
    bitrate= 3000000
    framerate= 29.97
    myFile=retrieval(width,height,bitrate,framerate)
    
    store(myFile)
    calc(len(myFile))

else:
    print "Invalid input"







