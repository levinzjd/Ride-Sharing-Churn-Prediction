import pandas as pd
import datetime


def load_data(path):
    df = pd.read_csv(path, parse_dates=['last_trip_date','signup_date'])
    # create target varible churn
    date_cutoff = datetime.date(2014, 6, 1)
    df['churn'] = (df.last_trip_date < date_cutoff).astype(int)
    return df


def data_processing(df):
    # drop irrelavant signup_date column
    # drop last_trip_date column to avoid data leakage
    df.drop(['last_trip_date','signup_date'],axis=1,inplace=True)

    # additional column to mark the missing values for rating columns
    # to keep the signals of missing values
    df['avg_rating_of_driver_isnull'] = df.avg_rating_of_driver.isnull().astype(int)
    df['avg_rating_by_driver_isnull'] = df.avg_rating_by_driver.isnull().astype(int)

    df.avg_rating_of_driver = df.avg_rating_of_driver.fillna(value=0)
    df.avg_rating_by_driver = df.avg_rating_by_driver.fillna(value=0)

    # dummify categorical and binary varibles
    dic1 = {True: 1, False: 0}
    df["luxury_car_user"] = df["luxury_car_user"].map(dic1)
    df['phone'].fillna('no_phone', inplace=True)
    city_dummy = pd.get_dummies(df['city'],drop_first=True)
    phone_dummy = pd.get_dummies(df['phone'],drop_first=True)
    df_dummy = pd.concat([df, city_dummy, phone_dummy], axis=1)
    df_dummy.drop(['city','phone'], axis=1,inplace=True)
    return df_dummy


if __name__ == '__main__':
    df = load_data('data/churn_train.csv')
    df = data_processing(df)
