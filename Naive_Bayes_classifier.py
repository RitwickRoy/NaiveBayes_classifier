# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 22:46:07 2020

@author: Ritwick
"""
#
#  Naive Bayes classifier:
#  Notes: This is a basic interactive implementation.
#  1. Features can be mixed categorical and continuous
#  3. Classification assumes categorical target variable in the final column
#  Future imporvements: 
#             1. introduce Laplace correction
#             2. introduce test train split 
#             3. introduce classification report
#
import pandas as pd
import numpy as np
import seaborn as sns
import random
import math
import statistics
#
#
#
class NaiveBayes:
    def __init__(self,df):
        self.df = df
        self.nrows = self.df.shape[0]
        self.ncols = self.df.shape[1]
        self.col_names = df.columns
        self.TargetV = self.df[self.col_names[self.ncols-1]]
        self.Classes = self.TargetV.unique() 
        self.NClasses = self.TargetV.nunique() 
        self.featureType = self.df.dtypes
        self.nCatFeatureMaxUnique = 0
        self.nCatFeatures = 0
        self.CatFeature = np.zeros([self.ncols-1,2],dtype=int)
        self.CatFeatureLab = np.empty([1,1],dtype=object)
        self.ClaPriorProb = np.zeros(self.NClasses,dtype=float)
        self.ClaCondProb = np.zeros([1,1,1], dtype=float)
        self.TestFeature = np.empty([1,1],dtype=object)
        self.Predict = np.zeros(self.NClasses, dtype=float)

#################################################################
#
#   Count the number of unique labels for each feature.
#   Setup mappings for label/value for each feature.
#   Mappings are used to lookup class conditional probabilies
#   during prediction phase.
#
#################################################################
    def CountDistinctLabels(self):
        self.nCatFeatures = 0

        for col in range(self.ncols-1):
            if (self.featureType[col] == 'object'):
                self.nCatFeatures += 1
                self.nCatFeatureMaxUnique = max(self.nCatFeatureMaxUnique, df.iloc[:,col].nunique())
            elif (self.featureType[col] == 'bool'):
                self.nCatFeatures += 1
                self.nCatFeatureMaxUnique = max(self.nCatFeatureMaxUnique, 2) 
            else:
                self.nCatFeatures += 1
                self.nCatFeatureMaxUnique = max(self.nCatFeatureMaxUnique, 2) 
#
#       resize the arrays based on number of unique labels for each feature
#

        self.CatFeatureLab.resize(self.nCatFeatures,self.nCatFeatureMaxUnique)
        self.ClaCondProb.resize(self.NClasses,self.nCatFeatures,self.nCatFeatureMaxUnique)
        self.TestFeature.resize(self.nCatFeatures)

        iCat = 0
        for col in range (self.ncols-1):
            if self.featureType[col] == 'object':
                L = self.df.iloc[:,col].unique()
                self.CatFeature[iCat,0] = col
                self.CatFeature[iCat,1] = self.df.iloc[:,col].nunique()
                for i in range(self.df.iloc[:,col].nunique()):
                    self.CatFeatureLab[iCat,i] = L[i]
                iCat += 1
            elif (self.featureType[col] == 'bool'):
                self.CatFeatureLab[iCat,0] = False
                self.CatFeatureLab[iCat,1] = True
                self.CatFeature[iCat,0] = col
                self.CatFeature[iCat,1] = 2
                iCat += 1                
            else:
                self.CatFeature[iCat,0] = col
                self.CatFeature[iCat,1] = 2
                iCat += 1    


#################################################################
#
#   Compute Prior Probability P(yi) for each target class
#
#################################################################
    def Class_PriorProb(self):
        for i in range(self.NClasses):
            cond = self.TargetV == self.Classes[i]
            Target = self.TargetV[cond]
            ni = Target.shape
            self.ClaPriorProb[i] = float(ni[0]/self.nrows)
        
#################################################################
#
#   Compute conditional class Probability P(xk|yi) for each feature
#   conditional probabilites are stored in a n X m X p tensor
#   n: number of Target Classes
#   m: number of features (categorical and continuous features)
#   p: maximum number of unique feature labels.
#
#################################################################
    def Class_CondProb(self):
        for Cla in range(self.NClasses):
            class_name = self.Classes[Cla]
            condCla = self.df[self.col_names[self.ncols-1]] == class_name
            df_cla = self.df[condCla]
            nTotalCla, ndummy = df_cla.shape
            for cat in range(self.nCatFeatures):
                cat_col = self.CatFeature[cat,0]
                columnName = self.col_names[cat_col]
                if self.featureType[cat_col] == 'object':
                    for lab in range(self.CatFeature[cat,1]):
                        condFeat = df_cla[columnName] == self.CatFeatureLab[cat,lab]
                        nCountFeat, ndummy = df_cla[condFeat].shape
                        prob = float(nCountFeat/nTotalCla)
                        self.ClaCondProb[Cla,cat_col,lab] = prob
                elif self.featureType[cat_col] == 'bool':
                    for lab in range(self.CatFeature[cat,1]):
                        condFeat = df_cla[columnName] == self.CatFeatureLab[cat,lab]
                        nCountFeat, ndummy = df_cla[condFeat].shape
                        prob = float(nCountFeat/nTotalCla)
                        self.ClaCondProb[Cla,cat_col,lab] = prob
                else:
                    mean = df_cla[columnName].mean()
                    sd = df_cla[columnName].std()
                    self.ClaCondProb[Cla,cat_col,0] = mean
                    self.ClaCondProb[Cla,cat_col,1] = sd
                    

#################################################################
#
#   Predict Class posterior probability
#   compute: P(yi|(x1,x2,...xn)) =  P(x1|yi)*P(x2|yi)...P(xn|yi)*P(yi)
#   The denominator term is not computed.
#   The posterior probabilities are normalized.
#
#################################################################
    def PredictClaPostProb(self):
        ProbMax = 0.0
        ProbMaxIdx = 0
        for Cla in range(self.NClasses):
            PriorProb = self.ClaPriorProb[Cla]
            PostProb = PriorProb
            for cat in range(self.nCatFeatures):
                cat_col = self.CatFeature[cat,0]    
                test_cat = self.TestFeature[cat]
                if self.featureType[cat_col] == 'object':
                    for lab in range(self.CatFeature[cat,1]):
                        if test_cat == self.CatFeatureLab[cat,lab]:
                            ClaCondProb = self.ClaCondProb[Cla,cat_col,lab]
                            PostProb = PostProb*ClaCondProb
                elif self.featureType[cat_col] == 'bool':
                    for lab in range(self.CatFeature[cat,1]):
                        if test_cat == self.CatFeatureLab[cat,lab]:
                            ClaCondProb = self.ClaCondProb[Cla,cat_col,lab]
                            PostProb = PostProb*ClaCondProb
                else:
                    mean = self.ClaCondProb[Cla,cat_col,0]
                    sd = self.ClaCondProb[Cla,cat_col,1]
                    ClaCondProb = self.NormalProb(test_cat,mean,sd)
                    PostProb = PostProb*ClaCondProb

            self.Predict[Cla] = PostProb
            if PostProb > ProbMax:
                ProbMax = PostProb
                ProbMaxIdx = Cla
        sumProb = np.sum(self.Predict)
        for Cla in range(self.NClasses):
            self.Predict[Cla] = self.Predict[Cla]/sumProb
        print('')
        print('Predicted Target Class:',self.Classes[ProbMaxIdx], 'Probability:',self.Predict[ProbMaxIdx])
        print('')
        
#################################################################
#
#   Compute probability for normal distribution
#   Assumption: The continuous feature is normally distributed
#
#################################################################
    def NormalProb(self,X, mean, sd):
        fact0 = 1.0/(sd*math.sqrt(2.0*np.pi))
        fact1 = -0.5*((X - mean)/sd)*((X - mean)/sd)
        Prob = fact0*math.exp(fact1)
        return Prob

#################################################################
#
#   Read Feature values for the new data-point.
#
#################################################################
    def ReadTestFeature(self):
        for cat in range(self.nCatFeatures):
            cat_col = self.CatFeature[cat,0]
            if self.featureType[cat_col] == 'object':
                print('Input value for Feature:',self.col_names[cat_col],' || Valid values:',self.CatFeatureLab[cat,:])
            elif self.featureType[cat_col] == 'bool':
                print('Input value for Feature:',self.col_names[cat_col],' || Valid values: False True')
            elif self.featureType[cat_col] == 'int64':
                print('Input value for Feature:',self.col_names[cat_col],' || Valid integer values')
            elif self.featureType[cat_col] == 'float':
                print('Input value for Feature:',self.col_names[cat_col],' || Valid float values')
            u_str = input('')
            if self.featureType[cat_col] == 'object':
                self.TestFeature[cat] = u_str
            elif self.featureType[cat_col] == 'bool':
                if u_str == 'False':
                    self.TestFeature[cat] = False
                elif u_str == 'True':
                    self.TestFeature[cat] = True
            else:
                if self.featureType[cat_col] == 'int64':
                    self.TestFeature[cat] = int(u_str)
                elif self.featureType[cat_col] == 'float':
                    self.TestFeature[cat] = float(u_str)

#################################################################
#
#   Test driver
#   1. Read Data-Set
#   2. compute prior probabilities for target classes
#   3. compute class conditional probabilities for features
#   4. make prediction for the given new data-point
#
#################################################################
u_str = input('enter filename: ')
df = pd.read_csv(u_str)  
#
nb_do = NaiveBayes(df)
nb_do.CountDistinctLabels()
nb_do.Class_PriorProb()
nb_do.Class_CondProb()
print()
print('#### Predict Target Class ####')
print()
Predict = True
while Predict == True:
    nb_do.ReadTestFeature()
    nb_do.PredictClaPostProb()
    u_str1 = input('Make another prediction (Yes/No) ?')
    if u_str1 == 'Yes':
        Predict = True
    else:
        Predict = False

