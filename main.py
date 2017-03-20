import numpy as np
import pandas as pd
import pylab
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

def sigmoid(s):
    return 1.0/(1.0+np.exp(-s))

def sigmoid_gradient(s):
    return sigmoid(s) * (1-sigmoid(s))

def train_nn(X,y,Theta1,Theta2,l,rate):
    print('train\n')
    m = len(X[:,1])
    X = np.append(np.ones(m).reshape(m,1),X,axis=1)
    z1 = sigmoid(np.dot(X,np.transpose(initial_Theta1)))
    z1 = np.append(np.ones(m).reshape(m,1),z1,axis=1)
    fz = np.dot(z1,np.transpose(initial_Theta2))

    J = 0
    for i in range(0,m):
        J+= y[i]*np.log(sigmoid(fz[i])) + (1-y[i])*np.log(sigmoid(fz[i]))
    J /= m

    #J = J +
    ###########################
    Del1 = np.zeros(shape=(len(Theta1[:,0]),len(Theta1[0,:])))
    Del2 = np.zeros(shape=(len(Theta2[:,0]),len(Theta2[0,:])))
    #Theta1_grad = np.zeros(np.initial_Theta1.size())
    #Theta2_grad = np.zeros(np.initial_Theta2.size())

    for i in range(0,m):

        #Step 1
        z = np.dot(initial_Theta1,np.transpose(X[i,:]))
        layer1 = sigmoid(z)
        layer1 = np.append(np.ones(1),layer1,axis=0)
        Olayer = sigmoid(np.dot(Theta2,layer1))
        #print(Olayer)

        #Step 2
        delta3 = abs(Olayer - y[i])

        #Step 3
        delta2 = np.dot(np.transpose(Theta2),delta3) * sigmoid_gradient(np.append(np.ones(1),z,axis=0))
        #print(delta2)
        delta2=delta2[1:]

        #Step4
        Del1 = Del1 + np.outer(delta2,X[i,:])
        #print(len(Del2[:,0]),len(Del2[0,:]))
        #print()
        Del2 = Del2 + delta3 * np.transpose(layer1)

    Theta1_grad = (Del1 + l * Theta1)/m
    Theta2_grad = (Del2 + l * Theta2)/m

    Theta1_grad[:,1] = Del1[:,1]/m
    Theta2_grad[:,1] = Del2[:,1]/m

    Theta1 = Theta1 - rate * Theta1_grad
    Theta2 = Theta2 - rate * Theta2_grad

    return [Theta1,Theta2,J]

def predict(Theta1,Theta2,X):
    m = len(X[:,1])
    h1 = sigmoid(np.dot(np.append(np.ones(m).reshape(m,1),X,axis=1), np.transpose(Theta1)))
    p = sigmoid(np.dot(np.append(np.ones(m).reshape(m,1),h1,axis=1), np.transpose(Theta2)))
    p = np.ceil(p - 0.5)
    return p


train_data = load_train_data()
y = train_data['Salary'].as_matrix()
del train_data['Salary']
X = train_data.as_matrix()
parameters = len(X[0,:])
m = len(X[:,1])
hidden_layer = 50
iterations = 20
l=0 #lambda
rate = 0.01

initial_Theta1 = np.random.rand(hidden_layer,parameters)
initial_Theta1 = np.append(np.ones(hidden_layer).reshape(hidden_layer,1),initial_Theta1,axis=1)
initial_Theta2 = np.random.rand(1,hidden_layer)
initial_Theta2 = np.append(np.ones(1).reshape(1,1),initial_Theta2,axis=1)

Cost = np.empty(0)
x = np.empty(0)
#pylab.plot(y,Cost)
for i in range(0,iterations):
    [initial_Theta1,initial_Theta2,J] = train_nn(X,y,initial_Theta1,initial_Theta2,l,rate)
    Cost = np.append(Cost,J)
    x = np.append(x,i)
    print(i,J)

print(Cost)

p=predict(initial_Theta1,initial_Theta2,X)

print('Accuracy ', (sum(p)/sum(y)))
#print(len(50),len(Cost))
pylab.plot(x,Cost)

