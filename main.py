import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import os

#Loading training set and filtering data
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

#Normalizing a parameter vector
def normalize(parameter):
    mean = np.mean(parameter,dtype=float)
    variance = np.std(parameter)
    new_parameter = (parameter - mean)/variance
    return [mean, variance, new_parameter]

#Sigmoid function
def sigmoid(s):
    return 1.0/(1.0+np.exp(-s))

#Derivative of Sigmoid function
def sigmoid_gradient(s):
    return sigmoid(s) * (1-sigmoid(s))

#Evaluation function given threshold of 0.5
def predict(Theta1,Theta2,X,y):
    m = len(X[:,1])
    hidden_layer_prediction = sigmoid(np.dot(X,theta1.T))
    hidden_layer_prediction = np.append(np.ones(m).reshape(m,1),hidden_layer_prediction,axis=1)
    h2 = np.dot(hidden_layer_prediction,theta2.T)
    threshold=0.5
    true_positives,false_positives,false_negatives=0,0,0
    for i in range(0,m):
        if(y[i]==1 and h2[i]>=threshold):
            true_positives+=1
        if(y[i]==1 and h2[i]<threshold):
            false_negatives+=1
        if(y[i]==0 and h2[i]>=threshold):
            false_positives+=1
    precision = true_positives/(true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    F1 = 2*precision*recall / (precision + recall)
    return F1


start_time = time.time()    ## Keeping the track of time

############## Importing data #####################
train_data = load_train_data()
y = train_data['Salary'].as_matrix()
del train_data['Salary']
X = train_data.as_matrix()

#Feature Normalization
[X0_mean , X0_variance, X[:,0]] = normalize(X[:,0])
[X1_mean , X1_variance, X[:,1]] = normalize(X[:,1])

parameters = len(X[0,:])
m = len(X[:,1])
hidden_layer = parameters
iterations = 3*m
l=0.1
rate = 0.03

X = np.append(np.ones(m).reshape(m,1),X,axis=1)

# Seeding for debugging purposes
np.random.seed(1)
theta1 = np.random.rand(hidden_layer,parameters+1)
theta2 = np.random.rand(1,hidden_layer+1)*0.01

print(predict(theta1,theta2,X,y))
#print(theta1.sum())
J_Arr = np.empty(0)
for iter in range(iterations):
    #print("iter: ",i)
    i = np.random.randint(0,m-1)
    x = X[i,:].T
    #print(x)
    #print(theta1.dot(x))
    l1 = sigmoid(theta1.dot(x))
    l1 = np.append(np.ones(1),l1,axis=0)
    #print(theta2.dot(l1))
    l2 = sigmoid(theta2.dot(l1))
    #print(l2)

    #Evaluating the cost function after every 1000th iteration
    if(i%1000 == 0):
        #print(theta1.sum())
        hidden_layer_prediction = sigmoid(np.dot(X,theta1.T))
        hidden_layer_prediction = np.append(np.ones(m).reshape(m,1),hidden_layer_prediction,axis=1)
        #print(hidden_layer_prediction)
        fz = np.dot(hidden_layer_prediction,theta2.T)
        #print(fz)
        J = 0
        for r in range(0,m):
            J+= y[r]*np.log(sigmoid(fz[r])) + (1-y[r])*np.log(1 - sigmoid(fz[r]))
        print("Cost: ", -J)
        J_Arr = np.append(J_Arr,-J)


    #E2 = (-y[i]/l2) - ((1-y[i])/(1-l2))
    E2 = l2 - y[i]
    w = theta2[0,1:] * (l1*(1-l1))[1:].T
    p = x.reshape(parameters+1,1).dot(w.reshape(hidden_layer,1).T)
    theta1 = theta1 - rate * E2 * l2*(1-l2) * p.T
    theta2 = theta2 - rate * E2 * l2*(1-l2) * l1.T
    #print(theta1.sum(),theta2.sum())

print(predict(theta1,theta2,X,y))
print ("time elapsed: ", format(time.time() - start_time))

plt.plot(J_Arr)
plt.show()
