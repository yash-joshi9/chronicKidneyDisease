# Importing libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest #Also known as Information Gain
from sklearn.feature_selection import chi2

dataset = pd.read_csv('kidney_disease.csv', sep = ',')

# To check column names
print (dataset.head())

# Abbreviation for columns
columns = pd.read_csv('data_description.txt', sep = '-')
print (columns)

# Resetting index and purifying the data
columns = columns.reset_index()
columns.columns = ['cols', 'abb_col_names']

# Renaming column names to meaningful names
dataset.columns = columns['abb_col_names'].values
print(columns)
print (dataset.head())

# Checking datatypes for the data
print(dataset.dtypes)

# Converting some required fields into respective datatypes
def convert_dtypes(dataset, feature):
    dataset[feature] = pd.to_numeric(dataset[feature],errors = 'coerce')

features = ['packed cell volume', 'white blood cell count', 'red blood cell count']

for feature in features:
    convert_dtypes(dataset,feature)

print (dataset.dtypes)

# Filtering out non - required field from the file

dataset.drop('id', axis = 1, inplace = True)
print (dataset.head())

# Extracting columns based upon its data type in categorical columns and numerical columns
def extract_cat_num(dataset):
    cat_col=[col for col in dataset.columns if dataset[col].dtype == 'object']
    num_col=[col for col in dataset.columns if dataset[col].dtype != 'object']
    return cat_col, num_col

cat_col, num_col = extract_cat_num(dataset)
print (extract_cat_num(dataset))

# Total unique categories in our categorical features to check if any dirtiness in data or not
for col in cat_col:
    print('{} has {} values '.format(col,dataset[col].unique()))
    print('\n')

# ckd-chronic kidney disease
# notckd-->> not crornic kidney disease

# So we need to correct 2 features and the target variable which contain certain discrepancy in some values.
# Replace incorrect values

dataset['diabetes mellitus'].replace(to_replace = {'\tno':'no','\tyes':'yes',' yes':'yes'},inplace=True)

dataset['coronary artery disease'] = dataset['coronary artery disease'].replace(to_replace = '\tno', value='no')

dataset['class'] = dataset['class'].replace(to_replace = 'ckd\t', value = 'ckd')

# Again printing to check any dirtiness left or not
for col in cat_col:
    print('{} has {} values '.format(col,dataset[col].unique()))
    print('\n')

# Printing number of numerical features
print (len(num_col))

# Checking Features distribution
plt.figure(figsize=(30,20))
for i,feature in enumerate(num_col):
    plt.subplot(5,3,i+1)
    dataset[feature].hist()
    plt.title(feature)

# Checking label distribution for categorial data
print (len(cat_col))
plt.figure(figsize=(20,20))
for i,feature in enumerate(cat_col):
    plt.subplot(4,3,i+1)
    sns.countplot(dataset[feature])

# A few features have imbalanced categories. Stratified folds will be necessary while cross validation.
sns.countplot(x='class',data=dataset)
plt.xlabel("class")
plt.ylabel("Count")
plt.title("target Class")

# Correlation
plt.figure(figsize=(10,8))
corr_dataset = dataset.corr()
sns.heatmap(corr_dataset,annot=True)

# Positive Correlation:

# Specific gravity -> Red blood cell count, Packed cell volume and Hemoglobin
# Sugar -> Blood glucose random
# Blood Urea -> Serum creatinine
# Hemoglobin -> Red Blood cell count <- packed cell volume


# Negative Correlation:
# Albumin, Blood urea -> Red blood cell count, packed cell volume, Hemoglobin
# Serum creatinine -> Sodium

dataset.groupby(['red blood cells','class'])['red blood cell count'].agg(['count','mean','median','min','max'])

# Let's check for Positive correlation and its impact on classes
px.violin(dataset,y='red blood cell count',x="class", color="class")

px.scatter(dataset,'haemoglobin','packed cell volume')

# Analysing distribution of 'red_blood_cell_count' in both Labels 

grid=sns.FacetGrid(dataset, hue="class",aspect=2)
grid.map(sns.kdeplot, 'red blood cell count')
grid.add_legend()

# Both distributions are quite different, distribution CKD is quite normal and evenly distributed but not CKD distribution is a little bit left-skewed but quite close to a normal distribution

# Defining violin and scatter plot & kde_plot functions
def violin(col):
    fig = px.violin(dataset, y=col, x="class", color="class", box=True)
    return fig.show()

def scatters(col1,col2):
    fig = px.scatter(dataset, x=col1, y=col2, color="class")
    return fig.show()

def kde_plot(feature):
    grid = sns.FacetGrid(dataset, hue="class",aspect=2)
    grid.map(sns.kdeplot, feature)
    grid.add_legend()

kde_plot('red blood cell count')
kde_plot('haemoglobin')

scatters('red blood cell count', 'packed cell volume')
scatters('red blood cell count', 'haemoglobin')
scatters('haemoglobin','packed cell volume')

# 1.RBC count range ~2 to <4.5 and Hemoglobin between 3 to <13 are mostly classified as positive for chronic kidney  
# disease(i.e ckd).
# 2.RBC count range >4.5 to ~6.1 and Hemoglobin between >13 to 17.8 are classified as negative for chronic kidney 
# disease(i.e nockd).

violin('red blood cell count')
violin('packed cell volume')

# Now let's check for negative correlation and its impact on classes

# Albumin, Blood urea -> Red blood cell count, packed cell volume, Haemoglobin

scatters('red blood cell count','albumin')

# Clearly, albumin levels of above 0 affect ckd largely
scatters('packed cell volume','blood urea')

# Packed cell volume >= 40 largely affects to be non ckd
fig = px.bar(dataset, x="specific gravity", y="packed cell volume",color='class', barmode='group',height=400)
fig.show()

# Clearly, specific gravity >=1.02 affects non ckd
dataset.head()
dataset.isna().sum().sort_values(ascending=False)

sns.countplot(dataset['red blood cells'])
data=dataset.copy()


# Random Value Imputation
data['red blood cells'].isnull().sum()

data['red blood cells'].dropna().sample()

random_sample=data['red blood cells'].dropna().sample(data['red blood cells'].isnull().sum())
print (random_sample)

print (random_sample.index)

data[data['red blood cells'].isnull()].index

random_sample.index=data[data['red blood cells'].isnull()].index

data.loc[data['red blood cells'].isnull(),'red blood cells']=random_sample

sns.countplot(data['red blood cells'])

print (data['red blood cells'].value_counts()/len(data))

print (len(dataset[dataset['red blood cells']=='normal'])/248)

print (len(dataset[dataset['red blood cells']=='abnormal'])/248)

# lets create a function so that it can be done easily for all features
def Random_value_imputation(feature):
    random_sample=data[feature].dropna().sample(data[feature].isnull().sum())               
    random_sample.index=data[data[feature].isnull()].index
    data.loc[data[feature].isnull(),feature]=random_sample

mode=data['pus cell clumps'].mode()[0]

print (mode) # it will otput not present values

data['pus cell clumps']=data['pus cell clumps'].fillna(mode)

def impute_mode(feature):
    mode=data[feature].mode()[0]
    data[feature]=data[feature].fillna(mode)

for col in cat_col:
    impute_mode(col)

# cleaning categorical features with missing values
print (data[cat_col].isnull().sum())

# cleaning numerical features with missing values
print (data[num_col].isnull().sum())

# lets fill missing values in Numerical features using Random value Imputation
for col in num_col:
    Random_value_imputation(col)

data[num_col].isnull().sum()

# feature Encoding
for col in cat_col:
    print('{} has {} categories'.format(col, data[col].nunique()))

# as we have just 2 categories in each feature then we can consider Label Encoder as it will not cause Curse of Dimensionality
le = LabelEncoder()

for col in cat_col:
    data[col]=le.fit_transform(data[col])

# Feature Importance
# SelectKBest-to select k best features
# chi2-Internally this class is going to check that whether p-value is less than 0.05 or not
# based on that,it will actually order all the features

ind_col=[col for col in data.columns if col!='class']
dep_col='class'

X=data[ind_col]
y=data[dep_col]

ordered_rank_features=SelectKBest(score_func=chi2,k=20)
ordered_feature=ordered_rank_features.fit(X,y)

print (ordered_feature)

#To get scores(rank) of feature,what we can do we can use scores function
ordered_feature.scores_

datascores=pd.DataFrame(ordered_feature.scores_,columns=["Score"])
datascores

dfcolumns=pd.DataFrame(X.columns)
dfcolumns

features_rank=pd.concat([dfcolumns,datascores],axis=1)

print (features_rank)

# Higher the score is,more important feature is
features_rank.columns=['Features','Score']
features_rank

# fetch largest 10 values of Score column
features_rank.nlargest(10,'Score')

selected_columns=features_rank.nlargest(10,'Score')['Features'].values

X_new=data[selected_columns]

# Model Building
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_new,y,train_size=0.75)

print(X_train.shape)
print(X_test.shape)

# check whether dataset is imbalance or not
y_train.value_counts()

# finding best model using Hyperparameter optimization
from xgboost import XGBClassifier
XGBClassifier()

# Hyper Parameter Optimization with respect to XGBoost

params={
 "learning_rate"    : [0.05, 0.20, 0.25 ] ,
 "max_depth"        : [ 5, 8, 10, 12],
 "min_child_weight" : [ 1, 3, 5, 7 ],
 "gamma"            : [ 0.0, 0.1, 0.2 , 0.4 ],
 "colsample_bytree" : [ 0.3, 0.4, 0.7 ]
    
}

from sklearn.model_selection import RandomizedSearchCV

from xgboost import XGBClassifier
classifier=XGBClassifier()

import warnings
from warnings import filterwarnings
filterwarnings('ignore')

random_search=RandomizedSearchCV(classifier,param_distributions=params,n_iter=5,scoring='roc_auc',n_jobs=-1,cv=5,verbose=3)

random_search.fit(X_train, y_train)

print (random_search.best_estimator_)

random_search.best_params_

classifier=XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=0.4, gamma=0.0, gpu_id=-1,
              importance_type='gain', interaction_constraints='',
              learning_rate=0.25, max_delta_step=0, max_depth=5,
              min_child_weight=1, monotone_constraints='()',
              n_estimators=100, n_jobs=2, num_parallel_tree=1,
              objective='binary:logistic', random_state=0, reg_alpha=0,
              reg_lambda=1, scale_pos_weight=1, subsample=1,
              tree_method='exact', use_label_encoder=True,
              validate_parameters=1, verbosity=None)

classifier.fit(X_train,y_train)

y_pred=classifier.predict(X_test)

from sklearn.metrics import confusion_matrix,accuracy_score
confusion = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:')
print(confusion)

plt.imshow(confusion)

print (accuracy_score(y_test, y_pred))