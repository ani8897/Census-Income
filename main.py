import numpy as np
import pandas as pd
import builtins
from collections import defaultdict
import os

def load_train_data():
    file_dir = os.path.dirname(__file__)
    file_path= os.path.join(file_dir,'data/train.csv')
    dataset = pd.read_csv(file_path)
    dataset.replace('?',np.nan, inplace=True)
    del dataset['id']

    ##############
    dataset.loc[(dataset['native-country']!='United-States')
                & (dataset['native-country'].notnull()),'native.country'] = 'Not United States'
    country_dummies = pd.get_dummies(dataset['native-country'], prefix='origin_country',
                                 drop_first=True,
                                dummy_na=True)
    '''
    def LogOr(x,y):
        if(isinstance(x,str)): return 0
        if(x>y):
            return x
        else:
            return 0
    country_dummies = country_dummies.apply(LogOr,axis=0)'''
    dataset = pd.concat([dataset, country_dummies], axis=1)
    '''
    def country(s):
        m = {'United-States' : 0}
        map = defaultdict(lambda : 1, m)
        return map.get(s)
    dataset['native.country']=dataset['native-country'].apply(country)
    #dataset = pd.concat([ dataset,nc ], axis=1)'''
    del dataset['native-country']
    del dataset['native.country']

    ################
    dataset.loc[dataset['capital-loss']>0, 'capital-loss'] = 1
    dataset.loc[dataset['capital-gain']>0, 'capital-gain'] = 1

    ##############
    sex_dummies = pd.get_dummies(dataset['sex'], prefix='', drop_first=True)
    dataset = pd.concat([dataset, sex_dummies], axis=1)
    del dataset['sex']

    ###############
    race_dict = {'Black': 'non_White', 'Asian-Pac-Islander': 'non_White',
             'Other': 'non_White', 'Amer-Indian-Eskimo': 'non_White'}

    race_dummies = pd.get_dummies(dataset['race'].replace(race_dict.keys(), race_dict.values()),
                              prefix='race', drop_first=True)
    dataset = pd.concat([dataset, race_dummies], axis=1)
    del dataset['race']

    ################
    occupy_dict = {'Armed-Forces': 'Protective-serv-Military', 'Protective-serv':
               'Protective-serv-Military'}

    occupy_dummies = pd.get_dummies(dataset['occupation'].replace(occupy_dict.keys(),
                                                          occupy_dict.values()),
                                                          prefix='occupation', drop_first=True,
                                                          dummy_na=True)
    dataset = pd.concat([dataset, occupy_dummies], axis=1)
    del dataset['occupation']

    ###################
    married_dict = {'Married-civ-spouse': 'Married', 'Married-spouse-absent': 'Married',
                'Married-AF-spouse': 'Married'}

    marital_dummies = pd.get_dummies(dataset['marital-status'].replace(married_dict.keys(),
                                                               married_dict.values()),
                                                               prefix='marital_status',
                                                               drop_first=True)
    dataset = pd.concat([dataset, marital_dummies], axis=1)
    del dataset['marital-status']

    ####################
    education_dict = {'1st-4th': 'Grade-school', '5th-6th': 'Grade-school', '7th-8th':
                  'Junior-high', '9th': 'HS-nongrad', '10th': 'HS-nongrad',
                  '11th': 'HS-nongrad', '12th': 'HS-nongrad', 'Masters':
                  'Graduate', 'Doctorate': 'Graduate', 'Preschool': 'Grade-school'}

    educ_dummies = pd.get_dummies(dataset.education.replace(education_dict.keys(),
                                                    education_dict.values()),
                                                    prefix='education',
                                                    drop_first=True)

    dataset = pd.concat([dataset, educ_dummies], axis=1)
    del dataset['education']

    #####################
    dataset.drop(dataset.loc[(dataset.workclass=='Without-pay') | (dataset.workclass=='Never-worked'), :].index,
        inplace=True)

    class_dict = {'Local-gov': 'Government', 'State-gov': 'Government', 'Federal-gov': 'Government',
              'Self-emp-not-inc': 'Self-employed', 'Self-emp-inc': 'Self-employed'}

    class_dummies= pd.get_dummies(dataset.workclass.replace(class_dict.keys(), class_dict.values()),
                              prefix='workclass', drop_first=True, dummy_na=True)

    dataset = pd.concat([dataset, class_dummies], axis=1)
    del dataset['workclass']

    ######################
    relate_dummies = pd.get_dummies(dataset.relationship, prefix='relationship', drop_first=True)

    dataset = pd.concat([dataset, relate_dummies], axis=1)
    del dataset['relationship']

    ########################
    dataset['hours-worked'] = np.nan
    dataset.loc[(dataset['hours-per-week']>=35) | (dataset['hours-per-week']<=40), 'hours-worked'] = 'Full_time'
    dataset.loc[dataset['hours-per-week']<35, 'hours-worked'] = 'Part_time'
    dataset.loc[dataset['hours-per-week']>40, 'hours-worked'] = '40+hrs'

    hours_dummies = pd.get_dummies(dataset['hours-worked'], prefix='WklyHrs', drop_first=True)

    dataset = pd.concat([dataset, hours_dummies], axis=1)

    del dataset['hours-per-week']
    del dataset['hours-worked']

    ##########################

    educ_dict = {1: '1-4', 2: '1-4', 3: '1-4', 4: '1-4', 5: '5-8', 6: '5-8',
             7: '5-8', 8: '5-8', 9: '9-12', 10: '9-12', 11: '9-12', 12: '9-12',
             13: '13-16', 14: '13-16', 15: '13-16', 16: '13-16'}

    educ_num = pd.get_dummies(dataset['education-num'].replace(educ_dict.keys(), educ_dict.values()),
                          prefix='YrsEduc', drop_first=False)
    dataset = pd.concat([dataset, educ_num], axis=1)

    del dataset['education-num']
    dataset['Salary'] = dataset['salary']
    del dataset['salary']
    return dataset

train_data = load_train_data().as_matrix()

#train_data.to_csv('stuff.csv')

print(train_data[0,1])

