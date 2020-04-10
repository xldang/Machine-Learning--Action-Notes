from Pipeline import *
from DIY_LR import *
import pandas as pd
import numpy as np

def fitlabel(x):
    if x=='no':return 0.0
    else:return 1.0
def fitDuration(x):
    if x<=100:return 100.0
    elif x>100 and x<=200:return 200.0
    elif x>200 and x<=300:return 300.0
    elif x>300 and x<=400:return 400.0
    elif x>400 and x<=500:return 500.0
    else:return 600.0
## Data reader
bank = pd.read_csv(r'C:\Users\lanti\Desktop\Library\ML data\classification\Bank customer classification\bank\bank-full.csv')
## Data pre shifting
bank['y']=bank['y'].apply(fitlabel)
## select 'duration' as shuffer feature. Then reduce its number of categeries.
bank['duration']=bank['duration'].apply(fitDuration)
## Feature tables (without the label 'y')
cat_attribs=['job','marital','education','default','housing','loan','contact','month','poutcome']
num_attribs=['age','balance','day','duration','campaign','pdays','previous']
## create a MyPipeline Object
obj=MyPipeline(bank,num_attribs,cat_attribs)
## Do shuffer split!
train,test=obj.shufferSplit('duration') # split set
## Preparing for a Pipeline job!
trainFeatures=train.drop('y',axis=1)
testFeatures=test.drop('y',axis=1)
trainLabel=train['y'].copy()
testLabel=test['y'].copy()
## Use my Pipeline
prepared_Train=obj.myPipUnion(trainFeatures)
prepared_Test=obj.myPipUnion(testFeatures)
trainLabel=np.array(trainLabel).reshape((trainLabel.shape[0],1))


myLR=LR()
weights=myLR.gradAscent1(prepared_Train,trainLabel)
myLR.plotMSE()

