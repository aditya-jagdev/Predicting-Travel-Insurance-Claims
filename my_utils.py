import numpy as np
import pandas as pd

import matplotlib
from matplotlib import colors as mcolors
from matplotlib import pyplot as plt
from matplotlib import ticker
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.model_selection import learning_curve


def plot_auc(y_test, y_pred):
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
    """Generate a simple plot of the test and training learning curve"""
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
    return np.log(value) if value > 0 else 0

def sqrt_transform(value):
    return np.sqrt(value)

def boxcox_transform(array):
    index_temp = list(array.index).copy()
    array_temp = array
    array_temp = np.where(array_temp == 0, 0.000000001, array_temp)
    return pd.Series(data=stats.boxcox(array_temp)[0],
                    index=index_temp) 

def check_skewness(array):
    sns.distplot(array)
    print(stats.skew(array))
    
def show_transformed(array):
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