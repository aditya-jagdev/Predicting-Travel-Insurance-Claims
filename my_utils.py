import gensim
import math
import nltk
import numpy as np
import operator
import pandas as pd
import pickle
import pydotplus
import random
import re
import seaborn as sns
import string
import sys
import time
import warnings
import zipfile

from collections import Counter

from gensim import corpora
from gensim.models.coherencemodel import CoherenceModel
from gensim.models.ldamodel import LdaModel
from gensim.models.lsimodel import LsiModel

from gensim.utils import simple_preprocess
from gensim.utils import simple_preprocess

from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import ClusterCentroids
from imblearn.under_sampling import TomekLinks
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE

from io import StringIO

from IPython.display import Image

import matplotlib
from matplotlib import pyplot as plt
from matplotlib import colors as mcolors
from matplotlib import ticker

from mlxtend.classifier import StackingClassifier

from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize

from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import *
from nltk.stem.wordnet import WordNetLemmatizer

from pprint import pprint

import pyLDAvis
import pyLDAvis.gensim

from scipy import stats

from scipy.cluster import hierarchy as sch

from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering

from sklearn.datasets import make_moons

from sklearn.decomposition import LatentDirichletAllocation
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import SelectPercentile

from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Lasso
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeClassifier


from sklearn.metrics import accuracy_score
from sklearn.metrics import auc
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import precision_score
from sklearn.metrics import precision_recall_fscore_support as error_metric
from sklearn.metrics import r2_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import scale
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import plot_tree

from string import punctuation

from textblob import TextBlob

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from wordcloud import WordCloud
from wordcloud import STOPWORDS

from xgboost import XGBClassifier
from xgboost import XGBRegressor

warnings.filterwarnings('ignore')
np.set_printoptions(threshold=sys.maxsize)


def plot_auc(y_test, y_pred):
    '''Uses the True and Predicted values to plot
    the AUROCC'''
    fpr, tpr, threshold = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(10,10))
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

def get_results_df(gscv_model,):
    '''Returns a dataframe containing the GridSearchCV
    results. Aids in sorting processing the results faster.'''
    # Creating a dictionary out of the GridSearchCV results
    results_dict = {}
    results = gscv_model.cv_results_
    
    for key in results.keys():
        if type(results[key]) == np.ndarray:
            results_dict[key] = list(results[key])
            
        elif type(results[key]) == np.ma.core.MaskedArray:
            results_dict[key] = list(results[key])
            
        elif type(results[key]) == list:
            pass
        else:
            print("Unkown type encountered")
        
    return_df = pd.DataFrame(data=results_dict,)
    
    if return_df.isnull().sum().max() > 0:
        print("Incorrect model training values have been omitted")
        print("{0} values".format(return_df.isnull().sum().max()))
    return_df.dropna(axis=0, inplace=True)
    
    return return_df

def get_scores(model, X_test, y_test,):
    '''Accepts a GridSearchCV trained model and prints the
    Accuracy, Recall, Precision, F1 Score, AUROCC, Confusion
    Matrix and Classification Report. Also plots the AUROCC'''
    y_pred = model.best_estimator_.predict(X_test,)
    print("Accuracy: {0}".format(accuracy_score(y_test, y_pred,)))
    print("\n")
    print("Recall: {0}".format(recall_score(y_test, y_pred,)))
    print("Precision: {0}".format(precision_score(y_test, y_pred,)))
    print("F1 Score: {0}".format(f1_score(y_test, y_pred,)))
    print("\n")
    print("ROC AUC: {0}".format(roc_auc_score(y_test, y_pred,)))
    print("\n")
    print("Confusion Matrix: \n{0}".format(confusion_matrix(y_test, y_pred,)))
    print("\n")
    print("Classification Report: \n{0}".format(classification_report(y_test, y_pred,)))
    plot_auc(y_test, y_pred)

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5)):
    '''Plots the learning curve for a model'''
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

def plot_learning_curve_new(estimator, title, X, y, axes=None, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    '''Plots the learning curve for a model'''
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes,
                       return_times=True, verbose=2)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, 'o-')
    axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
                         fit_times_mean + fit_times_std, alpha=0.1)
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model")

    # Plot fit_time vs score
    axes[2].grid()
    axes[2].plot(fit_times_mean, test_scores_mean, 'o-')
    axes[2].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1)
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Performance of the model")

    return plt

def check_null(df, **kwargs):
    '''Returns a dataframe which contains the number, % and bool of
    missing values in columns'''
    to_return = df.isnull().sum().to_frame()
    to_return.columns = ['Number_Missing']
    to_return['% Missing'] = (to_return['Number_Missing']/len(df))*100
    to_return['Missing?'] = df.isnull().any()
    
    print(f'Total Columns: {len(df)}')
    
    if 'omit' in kwargs:
        if kwargs['omit'] == False:
            return to_return.sort_values(by='Number_Missing', ascending=False)
    
    to_return = to_return[ to_return['Number_Missing'] > 0 ]
    
    if len(to_return) == 0:
        print('No missing values found.')
        return
    
    return to_return.sort_values(by='Number_Missing', ascending=False)

def attach_frequency(df, column):
    '''Takes a dataframe and column name as input. Returns the dataframe
    with an additional column which contains the freqeuncy of the values
    in the column'''
    frequency = df[column].value_counts().to_frame().reset_index()
    frequency.columns = [column, f'{column}_frequency']
    return pd.merge(left=df, right=frequency, on=column, how='left')

def check_recurring(df):
    '''Takes a dataframe as input and returns a dataframe containing
    the recurring values with their related statistics.'''
    check_df = pd.DataFrame()
    check_df['Column'] = df.columns
    
    unique = []
    for column in check_df['Column']:
        unique.append(df[column].nunique())
    check_df['Unique Values'] = unique
    
    not_null = []
    for column in check_df['Column']:
        not_null.append(len(df[df[column].notnull()]))
    check_df['Not Null Values'] = not_null
    
    check_df['Ratio to Total Number'] = check_df['Unique Values'] / check_df['Not Null Values']
    
    has_null = []
    for column in check_df['Column']:
        has_null.append(True if len(df[df[column].isnull()]) >= 1 else False)
    check_df['Has Null'] = has_null
    
    check_df.sort_values(by=['Has Null', 'Ratio to Total Number'], ascending=(False, True), inplace=True)
    
    return check_df

def set_x_tick_labels(multiples=10):
    '''A code snippet used for formatting the X axis
    ticker value occurences'''
    ax = plt.gca()
    if float(multiples).is_integer():
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
    else:
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
        
    ax.xaxis.set_major_locator(ticker.MultipleLocator(base=multiples))

def set_y_tick_labels(multiples=10):
    '''A code snippet used for formatting the Y axis
    ticker value occurences'''
    ax = plt.gca()
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%f'))
    ax.xaxis.set_major_locator(ticker.MultipleLocator(base=multiples))
    
def log_transform(value):
    '''Returns the log of a single value'''
    return np.log(value) if value > 0 else 0

def sqrt_transform(value):
    '''Returns the square root of a single value'''
    if value == 0:
        return 0
    elif value < 0:
        return -(np.sqrt(abs(value)))
    else:
        return np.sqrt(value)

def boxcox_transform(array):
    '''Accepts an array and returns the boxcox transformed array'''
    index_temp = list(array.index).copy()
    array_temp = array
    array_temp = np.where(array_temp == 0, 0.000000001, array_temp)
    return pd.Series(data=stats.boxcox(array_temp)[0],
                    index=index_temp) 

def check_skewness(array):
    '''Plots a distribution plot and prints the skewness'''
    sns.distplot(array)
    print(stats.skew(array))
    
def show_transformed(array):
    '''Applies Log, Sqrt and Boxcox transforms and prints
    the resulting data along with their plot and skew value.'''
    array_log = array
    array_log = array_log.apply(log_transform)
    print("Log skew: {0}".format(stats.skew(array_log)))
    sns.distplot(array_log)
    plt.show()
    
    array_sqrt = array
    array_sqrt = array_sqrt.apply(sqrt_transform)
    print("Sqrt skew: {0}".format(stats.skew(array_sqrt)))
    sns.distplot(array_sqrt)
    plt.show()
    
    array_boxcox = boxcox_transform(array)
    print("Boxcox skew: {0}".format(stats.skew(array_boxcox)))
    sns.distplot(array_boxcox)
    plt.show()
    
def scree_test(data):
    '''Plots the Scree Test curve'''
    A = data.copy()
    A = scale(A)
    num_vars = len(data.columns)
    num_obs = len(data)
    A = np.asmatrix(A.T) * np.asmatrix(A)
    U, S, V = np.linalg.svd(A) 
    eigvals = S**2 / np.sum(S**2)

    sing_vals = np.arange(num_vars) + 1
    plt.figure(figsize=(10, 10));
    plt.plot(sing_vals, eigvals);
    plt.title('Scree Plot');
    plt.xlabel('Principal Component');
    plt.ylabel('Eigenvalue');

    leg = plt.legend(['Eigenvalues from SVD'], loc='best', borderpad=0.3, 
                     shadow=False, prop=matplotlib.font_manager.FontProperties(size='small'),
                     markerscale=0.4)
    leg.get_frame().set_alpha(0.4);
    # leg.draggable(state=True)
    set_x_tick_labels(5);
    plt.show();
    
def explained_cumvar(data):
    '''Plots the explained cumulative variance curve'''
    x = data.copy()
    x = scale(x)

    covar_matrix = PCA(n_components = 151)

    covar_matrix.fit(x)
    variance = covar_matrix.explained_variance_ratio_ #calculate variance ratios

    var = np.cumsum( np.round( covar_matrix.explained_variance_ratio_, 
                              decimals = 3) * 100 )

    plt.figure(figsize=(10, 10));
    plt.plot(var);

    plt.xlabel('# of Features');
    plt.ylabel('% Variance Explained');

    plt.title('PCA Analysis');

    plt.style.context('seaborn-whitegrid');

    plt.grid();
    plt.show();
    
def get_clf_name(clf):
    '''Returns the name of a classifier'''
    end_string = str(clf).split('.')[-1].split("'")[0]
    capital_locations = [index for index, i in enumerate(end_string) if i.isupper()]
    processed_name = ""
    prev = 0
    for i in capital_locations:
        if i == 0:
            continue
        processed_name = processed_name + end_string[prev:i] + ' '
        prev = i
    processed_name = processed_name + end_string[i:]
    return processed_name

def get_all_cv(X_train__, y_train__, data_kind, clf_dict):
    '''Performs training with CV for given classifiers and
    plots their performances in terms of the "roc_auc" metric.'''
    cv_results = {}
    for clf in clf_dict.keys():
        print(f'Fitting {clf}')
        cv_results[ clf ] = cross_val_score(clf_dict[ clf ], X_train__, y=y_train__,
                                           scoring="roc_auc", cv=10,
                                           verbose=1, n_jobs=-1)
        time.sleep(1)

    cv_mean_metric = {}
    cv_std_metric = {}

    for cv_result in cv_results:
        cv_mean_metric[ cv_result ] = cv_results[ cv_result ].mean()
        cv_std_metric[ cv_result ] = cv_results[ cv_result ].std()

    df_cv_results = pd.DataFrame({"CV Mean Metric": [cv_mean_metric[i] 
                                                     for i in sorted(clf_dict.keys())],
                                  "CV Std Dev": [cv_std_metric[i]
                                                 for i in sorted(clf_dict.keys())],
                                  "Algorithm": sorted(clf_dict.keys()),
                                 })

    _ = plt.figure(figsize=(14, 7))
    _ = plt.grid()
    set_x_tick_labels(0.05) 
    figure = sns.barplot(x="CV Mean Metric", y="Algorithm", data=df_cv_results, 
                         palette="Set3", orient="h", 
                         **{'xerr': cv_std_metric[i]
                            for i in sorted(cv_std_metric.keys())},
                        )
    _ = figure.set_xlabel("Mean AUROC Curve")
    _ = figure.set_title(f'Cross validation scores for {data_kind}')
    _ = plt.show()