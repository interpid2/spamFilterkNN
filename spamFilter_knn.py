import matplotlib.pyplot as plt
import numpy as np
import re
from sklearn.metrics import confusion_matrix
from nltk import word_tokenize as tkn
from sklearn.neighbors import KNeighborsClassifier as KNN
from collections import Counter as Cnt

def drawcMatrix(trueSet, predictedSet):
    c_matrix=confusion_matrix(trueSet, predictedSet)
    norm_conf = []
    
    for i in c_matrix:
        a = 0
        tmp_arr = []
        a = sum(i, 0)
        for j in i:
            tmp_arr.append(float(j)/float(a))
        norm_conf.append(tmp_arr)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    res = ax.imshow(np.array(norm_conf), cmap=plt.cm.Greys, interpolation='nearest') 
 
    width = len(c_matrix)
    height = len(c_matrix[0]) 
 
    for x in xrange(width):
        for y in xrange(height):
            ax.annotate(str(c_matrix[x][y]), xy=(y, x),
                        horizontalalignment='center',
                        verticalalignment='center', color = 'green', size = 20)
    zbroj=np.sum(c_matrix)
    acc= (float(c_matrix[0,0]+c_matrix[1,1])/zbroj)*100
    mcr= 100-acc
    prec1= (float(c_matrix[0,0])/(np.sum(c_matrix[0])))*100
    prec2= (float(c_matrix[1,1])/(np.sum(c_matrix[1])))*100
    recall= (float(c_matrix[0,0])/(np.sum(c_matrix[:,0])))*100
    spec= (float(c_matrix[1,1])/(np.sum(c_matrix[:,1])))*100
    print "Accuracy: ", acc, "%"
    print "Missclasification rate: " , mcr, "%"
    print "Precision - class 0:",  prec1, "%"
    print "          - class 1:",  prec2, "%"
    print "Recall: ",  recall, "%"
    print "Specificity: ",  spec, "%"
    
def knnTrain(f, tPer=0.7, noNeigh=3):
    velTrain=int(len(f)*tPer)
    set_Train = f[:velTrain]
    set_test = f[velTrain:]
    knn=KNN(n_neighbors=noNeigh)
    knn.fit(set_Train[:,0:5],set_Train[:,5])
    trainCnt=Cnt(set_Train[:,5]).items()
    testCnt=Cnt(set_test[:,5]).items()
    print 'Train set sum (ham+spam=uk): ', trainCnt[0][1],"+",trainCnt[1][1],"=",velTrain
    print 'Test set sum (ham+spam=uk): ', testCnt[0][1],"+",testCnt[1][1],"=", len(f)-velTrain
    return set_Train, set_test, knn

def importAndPreproces(dat):
    f=open(dat,'r') 
    MailStats=np.zeros((1,6)) 
    for line in f:
        temp=line.lower().decode('latin1')
        last=len(tkn(temp)[-1])
        noChar=(len(line)-last-1)
        noSMS=np.ceil(noChar/160.0)
        noNum=sum([len(word) for word in tkn(temp) if re.match(r'.*[0-9]+.*',word)]) #za s
        no_nonAlpNum=0
        unknown=0;
        for i in xrange(noChar):
            if not temp[i].isalnum() and not temp[i].isspace():
                no_nonAlpNum=no_nonAlpNum+1
            if temp[i]==u'\xa3':
                unknown=1
        MailStats=np.vstack((MailStats,[noChar/(160.0*noSMS),
                             float(no_nonAlpNum)/noChar,
                             float(noNum)/noChar,
                             unknown,
                             noSMS,
                             last-3]))
        
    return np.random.permutation(MailStats[1:,:])

np.random.seed(5)
mail_all = importAndPreproces("english_big.txt")
              
trainSet, testSet, klasifikator = knnTrain(mail_all)
result=klasifikator.predict(testSet[:,0:5])
drawcMatrix(testSet[:,5],result)
