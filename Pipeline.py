from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelBinarizer
from sklearn.pipeline import FeatureUnion
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedShuffleSplit
import pandas as pd
import numpy as np

class MyLabelBinarizer(TransformerMixin):
    def __init__(self, *args, **kwargs):
        self.encoder = LabelBinarizer(*args, **kwargs)
    def fit(self, x, y=0):
        self.encoder.fit(x)
        return self
    def transform(self, x, y=0):
        return self.encoder.transform(x)
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.attribute_names].values


class MyPipeline():
    def __init__(self, inputData, num_attribs, cat_attribs):
        self.inputData = inputData
        self.num_attribs = num_attribs
        self.cat_attribs = cat_attribs

    def shufferSplit(self, feature='', n=1, size=0.2, state=42):
        '''You need to know which feature is the Mr.right'''
        split_obj = StratifiedShuffleSplit(n_splits=n, test_size=size, random_state=state)
        for train_index, test_index in split_obj.split(self.inputData, self.inputData[feature]):
            strat_train_set = self.inputData.loc[train_index]
            strat_test_set = self.inputData.loc[test_index]
            print('Original Data set shape:', self.inputData.shape)
            print("After ShuffleSplit,Train Set shape:", strat_train_set.shape, 'Test Set shape:', strat_test_set.shape)
        return strat_train_set, strat_test_set

    def x_numPip(self, data):
        # pipeline here
        num_pipeline = Pipeline(
            [('selector', DataFrameSelector(self.num_attribs)), ('imputer', SimpleImputer(strategy='median')),
             ('std_scaler', StandardScaler()), ])
        prepared = num_pipeline.fit_transform(data)
        return prepared

    def x_catPip(self, data):
        first_cat = self.cat_attribs[0]
        cat_pipeline = Pipeline([('selector', DataFrameSelector(first_cat)), ('laber_binarizer', MyLabelBinarizer()), ])
        prepared = cat_pipeline.fit_transform(data)
        if len(self.cat_attribs) == 1: return prepared
        for i in range(1, len(self.cat_attribs)):
            cat_pipeline = Pipeline(
                [('selector', DataFrameSelector(self.cat_attribs[i])), ('laber_binarizer', MyLabelBinarizer()), ])
            tmp = cat_pipeline.fit_transform(data)
            prepared = np.hstack((prepared, tmp))
        return prepared

    def myPipUnion(self, data):
        union = np.hstack((self.x_numPip(data), self.x_catPip(data)))
        print('Before Pipline,Input Data Shap is:', data.shape)
        print('After Pipline,Output Data Shap is:', union.shape)
        return union


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


# Try a Decision Tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_squared_error
tree_reg=DecisionTreeClassifier()
tree_reg.fit(prepared_Train,trainLabel)
predictions=tree_reg.predict(prepared_Test)
tree_mse=mean_squared_error(testLabel,predictions)
tree_rmse=np.sqrt(tree_mse)
print(tree_rmse)