import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix, roc_curve
from sklearn.model_selection import learning_curve, cross_validate
from sklearn.preprocessing import PolynomialFeatures
from sklearn.utils import class_weight
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import fbeta_score, make_scorer

def OHE_preprocessing(xtrain, xtest):
    tranformer = make_column_transformer((OneHotEncoder(drop = 'if_binary'), make_column_selector(dtype_include=object)),
                                          remainder='passthrough',  verbose_feature_names_out=False)
    tranformer.fit(xtrain)
    index_train = xtrain.index
    index_test = xtest.index
    features = tranformer.get_feature_names_out()
    xtrain_ = pd.DataFrame(tranformer.transform(xtrain), index=index_train, columns=features)
    xtest_ = pd.DataFrame(tranformer.transform(xtest), index=index_test, columns=features)    
    return xtrain_, xtest_

# imputation des valeurs catégorielles
def impute_object(xtrain, xtest):
    tranformer = make_column_transformer((SimpleImputer(strategy='most_frequent'), make_column_selector(dtype_include=object)),
                                          remainder='passthrough',  verbose_feature_names_out=False)
    tranformer.fit(xtrain)
    index_train = xtrain.index
    index_test = xtest.index
    features = tranformer.get_feature_names_out()
    xtrain_ = pd.DataFrame(tranformer.transform(xtrain), index=index_train, columns=features)
    xtest_ = pd.DataFrame(tranformer.transform(xtest), index=index_test, columns=features)    
    return xtrain_, xtest_

# imputer la variable EXT_SOURCE_1 par la regression de la variable age
def impute_ext_source_1(X):
    train = X.loc[X.EXT_SOURCE_1.notna(),['age', 'EXT_SOURCE_1']]
    xtrain = train.age.array.reshape(-1,1)
    ytrain = train.EXT_SOURCE_1.array.reshape(-1,1)
    xtest = X.loc[X.EXT_SOURCE_1.isna(),'age']
    lr = LinearRegression()
    lr.fit(xtrain,ytrain)
    impute = lr.intercept_[0] + xtest * lr.coef_[0]
    df = X.copy()
    df.EXT_SOURCE_1.fillna(impute, inplace=True)    
    return df

# imputation avec la médian
def impute_number_med(xtrain, xtest):
    tranformer = make_column_transformer((SimpleImputer(strategy='median'), make_column_selector(dtype_include='number')),
                                         remainder='passthrough',  verbose_feature_names_out=False)    
    tranformer.fit(xtrain)
    index_train = xtrain.index
    index_test = xtest.index
    features = tranformer.get_feature_names_out()
    xtrain_ = pd.DataFrame(tranformer.transform(xtrain), index=index_train, columns=features)
    xtest_ = pd.DataFrame(tranformer.transform(xtest), index=index_test, columns=features)    
    return xtrain_, xtest_

# OneHotEncoding et imputation avec la médian
def OHE_med_preprocessing(xtrain, xtest):
    tranformer = make_column_transformer((OneHotEncoder(drop = 'if_binary'), make_column_selector(dtype_include=object)),
                                         (SimpleImputer(strategy='median'), make_column_selector(dtype_include='number')),
                                          remainder='passthrough',  verbose_feature_names_out=False)
    tranformer.fit(xtrain)
    index_train = xtrain.index
    index_test = xtest.index
    features = tranformer.get_feature_names_out()
    xtrain_ = pd.DataFrame(tranformer.transform(xtrain), index=index_train, columns=features)
    xtest_ = pd.DataFrame(tranformer.transform(xtest), index=index_test, columns=features)    
    return xtrain_, xtest_
# std dataframe
def std_scale(xtrain,xtest):
    std = StandardScaler()
    std.fit(xtrain)
    index_train = xtrain.index
    index_test = xtest.index
    features = std.get_feature_names_out()
    xtrain_std = pd.DataFrame(std.transform(xtrain), index=index_train, columns=features)
    xtest_std = pd.DataFrame(std.transform(xtest), index=index_test, columns=features)
    return xtrain_std, xtest_std

# minmax scaler
def minmax_scale(xtrain, xtest):
    minmax = MinMaxScaler()
    minmax.fit(xtrain)
    index_train = xtrain.index
    index_test = xtest.index
    features = minmax.get_feature_names_out()
    xtrain_minmax = pd.DataFrame(minmax.transform(xtrain), index=index_train, columns=features)
    xtest_minmax = pd.DataFrame(minmax.transform(xtest), index=index_test, columns=features)    
    return xtrain_minmax, xtest_minmax

# OneHotEncoding et imputation avec la médian et std scaler
def OHE_med_preprocessing_std(xtrain, xtest):
    xtrain_, xtest_ = OHE_med_preprocessing(xtrain, xtest)
    xtrain_std, xtest_std = std_scale(xtrain_, xtest_)
    return xtrain_std, xtest_std
    
# OneHotEncoding et imputation avec la median et maxmin scaler
def OHE_med_preprocessing_minmax(xtrain, xtest):
    xtrain_, xtest_ = OHE_med_preprocessing(xtrain, xtest)
    xtrain_minmax, xtest_minmax = minmax_scale(xtrain_, xtest_)   
    return xtrain_minmax, xtest_minmax

# Polynomial features 

def poly_features(xtrain, xtest, features):
    poly = PolynomialFeatures(degree=3, include_bias=False)
    poly.fit(xtrain[features])
    poly_features = poly.get_feature_names_out()
    
    index_train = xtrain.index
    index_test = xtest.index
    
    poly_xtrain_array = poly.transform(xtrain[features])
    poly_xtest_array = poly.transform(xtest[features])

    poly_xtrain = pd.DataFrame(poly_xtrain_array, columns=poly_features, index=index_train)
    poly_xtest = pd.DataFrame(poly_xtest_array, columns=poly_features, index=index_test)
    
    poly_xtrain_all = pd.concat([poly_xtrain, xtrain.drop(columns= features)], axis=1)
    poly_xtest_all = pd.concat([poly_xtest, xtest.drop(columns= features)], axis=1)
    
    return poly_xtrain_all, poly_xtest_all





# apprentissage du modèle 
def learning_model(model, xtrain, ytrain, xtest, ytest, threshold=0.5, RocCurve=True, LearningCurve=False, w = None):
    pd.options.display.float_format = '{:.2f}'.format
    
    # definition du metrique de validation
    beta = 2
    f2_scorer = make_scorer(fbeta_score, beta=beta)    
    print(str(model),'\n')
    if w == None :
        model.fit(xtrain, ytrain)
        crossval = cross_validate(model, xtrain, ytrain, scoring=f2_scorer, cv=4, return_train_score=True)
    else : 
        weights = class_weight.compute_sample_weight(class_weight=w,y=ytrain)
        model.fit(xtrain, ytrain, sample_weight=weights)
        crossval = cross_validate(model, xtrain, ytrain, scoring=f2_scorer,cv=4, 
                                                             fit_params={'sample_weight': weights},return_train_score=True)
        
    print('F2 train  mean : ', round(crossval['train_score'].mean(),2))
    print('F2 validation mean : ', round(crossval['test_score'].mean(),2))
    print('\n')
        
    yprob_train = model.predict_proba(xtrain)[:,1]
    yprob_test = model.predict_proba(xtest)[:,1]
    ypred = yprob_test>threshold
    confusion_m = pd.DataFrame(confusion_matrix(ytest, ypred)).T
    confusion_m['total'] = confusion_m.sum(axis=1)
    confusion_m.loc['total',:] = confusion_m.sum(axis=0)
    print("confusion_matrix :")
    display(confusion_m.astype('int64').rename_axis(index = ['Actual'], columns=['predicted']))
    print(pd.DataFrame(precision_recall_fscore_support(ytest, ypred, beta=beta), columns=['0','1'], 
                       index=[ 'precision','recall', 'fbeta', 'support']).T)
    

#   learning curve
    if LearningCurve == True:
        
        if w == None :
            N, train_score, test_score = learning_curve(model, xtrain, ytrain, scoring=f2_scorer, cv=4, train_sizes=np.linspace(0.2,1,4))
        else : 
            N, train_score, test_score = learning_curve(model, xtrain, ytrain, scoring=f2_scorer, cv=4, train_sizes=np.linspace(0.2,1,4),
                                                        fit_params={'sample_weight': weights})
            
        plt.plot(N, train_score.mean(axis=1), label = 'Train score')
        plt.plot(N, test_score.mean(axis=1), 'r', label='Validation score')
        plt.title('Learning Curve')
        plt.xlabel('Nombre de clients')
        plt.ylabel('F2_score')
        plt.legend()
        plt.show()
        
#   roc curve
    if RocCurve == True :
        fpr_test, tpr_test, thld_test = roc_curve(ytest, yprob_test)
        plt.plot(fpr_test, tpr_test, label='test')
        plt.legend()
        plt.title('Courbe ROC')
        plt.xlabel('Taux de faux positif')
        plt.ylabel('Taux de vrai positif')
        plt.show() 
                  
    return model