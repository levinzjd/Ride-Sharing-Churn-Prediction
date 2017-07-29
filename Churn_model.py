import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from main import load_data, data_processing
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, make_scorer, recall_score, confusion_matrix, precision_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler



def cross_val(model,X,y,cv):
    f1 = make_scorer(f1_score)
    scores = cross_val_score(model,X,y,scoring=f1,cv=cv)
    return scores

def base_design_matrix(df):
    '''Baseline model with all available features'''
    y = df.pop('churn').values
    # capture the extreme usage behavior
    df['extreme_weekday_usage'] = ((df['weekday_pct'] == 0)|(df['weekday_pct'] == 100)).astype(int)
    df['I am rich'] = (df['surge_pct'] == 100).astype(int)

    # extract the low rating customer and driver
    df['unhappy_customer'] = ((df['avg_rating_of_driver'] > 0) & (df['avg_rating_of_driver'] < 4)).astype(int)
    df['unhappy_driver'] = ((df['avg_rating_of_driver'] > 0) & (df['avg_rating_of_driver'] < 4)).astype(int)

    X = df.values
    X_train,X_test,y_train,y_test = train_test_split(X,y,random_state = 0)
    # print 'Data imbalance check: ',np.mean(y)
    return X_train,X_test,y_train,y_test, df.columns

def standardize_col(df,col):
    sc = StandardScaler()
    df[col+'_std'] = sc.fit_transform(df[col])
    return df,sc

def design_matrix(df,test=False):
    #feature selection design matrix
    y = df.pop('churn').values
    df['extreme_weekday_usage'] = ((df['weekday_pct'] == 0)|(df['weekday_pct'] == 100)).astype(int)
    df['I am rich'] = (df['surge_pct'] == 100).astype(int)

    #Column to be used
    cols = ['avg_rating_by_driver_isnull','avg_rating_of_driver_isnull',"King's Landing",\
    'luxury_car_user','avg_surge','Winterfell','iPhone','extreme_weekday_usage','I am rich']
    X = df[cols].values
    if test:
        return X,y,cols
    else:
        X_train,X_test,y_train,y_test = train_test_split(X,y,random_state = 0)
        return X_train,X_test,y_train,y_test,cols

def lr_search(X_train,X_test,y_train,y_test,cols):
    lr = LogisticRegression()
    valid_scores = cross_val(lr,X_train,y_train,5)
    lr.fit(X_train,y_train)
    coef = lr.coef_
    pred_y = lr.predict(X_test)
    print '-'*50
    print 'cross_valid_f1_scores: {}, avg_valid_score: {}'.format(valid_scores,np.mean(valid_scores))
    print 'validation accuracy score:{}'.format(lr.score(X_test,y_test))
    print 'validation precision score:{}'.format(precision_score(y_test,pred_y))
    print 'validation f1 score:{}'.format(f1_score(y_test,pred_y))
    print 'validation recall score:{}'.format(recall_score(y_test,pred_y))
    print '-'*50
    print 'confusion_matrix'
    print confusion_matrix(y_test,pred_y)
    # print len(cols), len(coef[0])
    abs_value = np.abs(coef[0])

    # Print out absolute coefficient in descending order
    for i,j,k in sorted(zip(abs_value,coef[0],cols))[::-1]:
        print k+': ', j

    return lr


def test_final_model(model):
    test_data = load_data('data/churn_test.csv')
    test_data = data_processing(test_data)

    X,y,cols = design_matrix(test_data,True)
    test_pred = model.predict(X)
    print '='*50
    print 'confusion_matrix\n',confusion_matrix(y,test_pred)
    print 'test data f1 score: ',f1_score(y,test_pred)
    print 'test data precision score: ',precision_score(y,test_pred)
    print 'test data recall score: ',recall_score(y,test_pred)

def plot_distribution_by_churn(df,col):
    '''feature values distribution vs churn'''
    plt.hist(df[col][df['churn'] == 0],bins=20,alpha=.3,label='not churn')
    plt.hist(df[col][df['churn'] == 1],bins=20,alpha=.3,label='churn')
    plt.xlabel(col)
    plt.ylabel('# of users')
    plt.legend()
    plt.savefig('images/{}.png'.format(col))
    plt.show()
    plt.clf()

def plot_dists(df):
    for col in ['avg_surge','surge_pct','avg_dist','avg_rating_by_driver','avg_rating_of_driver',\
    'trips_in_first_30_days','weekday_pct']:
        plot_distribution_by_churn(df,col)

def plot_category(df,col):
    print pd.crosstab(df['churn'],df[col])

def plot_cats(df):
    print '-'*50
    plot_category(df,'city')
    print '-'*50
    plot_category(df,'phone')
    print '-'*50
    plot_category(df,'luxury_car_user')
    print '-'*50


if __name__ == '__main__':
    df = load_data('data/churn_train.csv')
    df = data_processing(df)
    # plot_category(df,'avg_rating_by_driver_isnull')
    # plot_category(df,'avg_rating_of_driver_isnull')
    # plot_dists(df)
    X_train,X_test,y_train,y_test,cols = design_matrix(df)
    model = lr_search(X_train,X_test,y_train,y_test,cols)
    test_final_model(model)
