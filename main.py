import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import floor
import time
import os

#Loading training set and filtering data
def load_data(fname):
    file_dir = os.path.dirname(__file__)
    file_path= os.path.join(file_dir,'data/'+fname+'.csv')
    dataset = pd.read_csv(file_path)
    dataset.replace('?','dum', inplace=True)
    id = dataset['id'].as_matrix()
    del dataset['id']

    ##############
    dataset.loc[(dataset['native-country']!='United-States')
                & (dataset['native-country'].notnull()),'native.country'] = 'Not United States'
    country_dummies = pd.get_dummies(dataset['native-country'],
                                 drop_first=True,
                                dummy_na=True)
    dataset = pd.concat([dataset, country_dummies], axis=1)
    del dataset['native-country']
    del dataset['native.country']
    del dataset[' Cambodia']
    del dataset[' Canada']
    del dataset[' China']
    del dataset[' Columbia']
    del dataset[' Cuba']
    del dataset[' Dominican-Republic']
    del dataset[' Ecuador']
    del dataset[' India']
    del dataset[' El-Salvador']
    del dataset[' England']
    del dataset[' France']
    del dataset[' Germany']
    del dataset[' Greece']
    del dataset[' Guatemala']
    del dataset[' Haiti']
    if(fname=='train'): del dataset[' Holand-Netherlands']
    del dataset[' Honduras']
    del dataset[' Hong']
    del dataset[' Hungary']
    del dataset[' Iran']
    del dataset[' Ireland']
    del dataset[' Italy']
    del dataset[' Jamaica']
    del dataset[' Japan']
    del dataset[' Laos']
    del dataset[' Mexico']
    del dataset[' Nicaragua']
    del dataset[' Outlying-US(Guam-USVI-etc)']
    del dataset[' Peru']
    del dataset[' Portugal']
    del dataset[' Poland']
    del dataset[' Philippines']
    del dataset[' Puerto-Rico']
    del dataset[' Scotland']
    del dataset[' South']
    del dataset[' Taiwan']
    del dataset[' Thailand']
    del dataset[' Trinadad&Tobago']
    del dataset[' Vietnam']
    del dataset[' Yugoslavia']




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
    if(fname=='train'):
        dataset['Salary'] = dataset['salary']
        del dataset['salary']
    return [dataset,id]

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
def predict(theta1,theta2,X,y):
    m = len(X[:,1])
    hidden_layer_prediction = sigmoid(np.dot(X,theta1.T))
    hidden_layer_prediction = np.append(np.ones(m).reshape(m,1),hidden_layer_prediction,axis=1)
    h2 = np.dot(hidden_layer_prediction,theta2.T)
    h2 = sigmoid(h2)
    threshold=0.3
    true_positives,false_positives,false_negatives=0,0,0
    for i in range(0,m):
        if(y[i]==1 and h2[i]>=threshold):
            true_positives+=1
        if(y[i]==1 and h2[i]<threshold):
            false_negatives+=1
        if(y[i]==0 and h2[i]>=threshold):
            false_positives+=1
    print(true_positives,false_positives,false_negatives)
    if(true_positives==0): return [0,0,0]
    precision = true_positives/(true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    F1 = 2*precision*recall / (precision + recall)
    return [precision,recall,F1]

def output(string,data,id):
    m=data.size
    data = np.squeeze(np.asarray(data))
    outdat = np.column_stack((id.flatten(),data.flatten()))
    #final_data = np.append(['ID','MEDV'],final_data,axis=0)
    np.savetxt(string,outdat,delimiter=',',header="id,salary", comments='')


start_time = time.time()    ## Keeping the track of time

############## Importing data #####################
[train_data,dump] = load_data('train')
train_data.drop(train_data.columns[[5,7,8,9,19,21,25,26,34,47,48,49,51,52,53,54,55,65,64,63]],axis =1,inplace=True)
train_data.to_csv('stuff.csv')
y = train_data['Salary'].as_matrix()
del train_data['Salary']
X_old_data = train_data.as_matrix()
m = len(X_old_data[:,1])
X_data = np.zeros(13*m).reshape(m,13)
X_data[:,0:7]=X_old_data[:,0:7]
X_data[:,12]=X_old_data[:,46]
for i in range(m):
    occMap = {7:-2,8:-6,9:-1,10:-3,11:-5,12:-4,13:6,14:3,15:2,16:1,17:4,18:5}
    for j in range(7,19):
        if(X_old_data[i,j]==1): X_data[i,7] = occMap[j]

    marstatMap = {19:1,20:-3,21:-1,22:-2,23:2}
    for j in range(19,24):
        if(X_old_data[i,j]==1): X_data[i,8] = marstatMap[j]

    eduMap = {24:-3,25:-3,26:-3,27:-3,28:-3,29:-4,
              30:-5,31:-1,32:-6,33:-1,34:3,35:5,36:4,37:2}
    for j in range(24,38):
        if(X_old_data[i,j]==1): X_data[i,9] = eduMap[j]

    relMap = {39:1,40:-2,41:-1,42:2,43:3}
    for j in range(39,44):
        if(X_old_data[i,j]==1): X_data[i,10] = relMap[j]

    partMap = {44:2,45:1}
    for j in range(44,46):
        if(X_old_data[i,j]==1): X_data[i,11] = partMap[j]

#np.savetxt('stuff2.csv',X_data,delimiter=',')

#Feature Normalization
[X0_mean , X0_variance, X_data[:,0]] = normalize(X_data[:,0])
[X1_mean , X1_variance, X_data[:,1]] = normalize(X_data[:,1])
[X1_mean , X1_variance, X_data[:,7]] = normalize(X_data[:,7])
[X1_mean , X1_variance, X_data[:,8]] = normalize(X_data[:,8])
[X1_mean , X1_variance, X_data[:,9]] = normalize(X_data[:,9])
[X1_mean , X1_variance, X_data[:,10]] = normalize(X_data[:,10])



parameters = len(X_data[0,:])
hidden_layer = parameters
iterations = 200*m
l=0
rate = 0.005

X_data = np.append(np.ones(m).reshape(m,1),X_data,axis=1)

partition = floor(3*m/4)
X = X_data[0:partition-1,:]
X_CV = X_data[partition:,:]
m = len(X[:,1])

################################# Logistic Regression ############################

############################### Neural Network ###################################

# Seeding for debugging purposes
np.random.seed(3)
theta1 = np.random.rand(hidden_layer,parameters+1)
theta2 = np.random.rand(1,hidden_layer+1)

print("Train : ",predict(theta1,theta2,X,y),"Cross Validate : ",predict(theta1,theta2,X_CV,y))
J_Arr,theta1_Arr,theta2_Arr,k = np.empty(0),np.empty(0),np.empty(0),0
t_precision_Arr, t_recall_Arr, t_F1_Arr = np.empty(0),np.empty(0),np.empty(0)
c_precision_Arr, c_recall_Arr, c_F1_Arr = np.empty(0),np.empty(0),np.empty(0)
v1,g1,v2,g2 = 0,0,0,0
for it in range(iterations):
    i = np.random.randint(0,m-1)
    #i = it%m
    x = X[i,:].T
    #print(x)
    l1 = sigmoid(theta1.dot(x))
    l1 = np.append(np.ones(1),l1,axis=0)
    #print(theta1.dot(x))
    l2 = sigmoid(theta2.dot(l1))
    #print(theta2.dot(l1),l2)
    E2 = l2 - y[i]
    w = theta2[0,1:] * (l1*(1-l1))[1:].T
    p = x.reshape(parameters+1,1).dot(w.reshape(hidden_layer,1).T)

    v1 = g1*v1 + rate * E2 * l2*(1-l2) * p.T + l*(theta1)/m
    v2 = g2*v2 + rate * E2 * l2*(1-l2) * l1.T + l*(theta2)/m
    theta1 = theta1 - v1
    theta2 = theta2 - v2

    #Evaluating the cost function after every 1000th iteration
    if(i%38000 == 0):
        hidden_layer_prediction = sigmoid(np.dot(X,theta1.T))
        hidden_layer_prediction = np.append(np.ones(m).reshape(m,1),hidden_layer_prediction,axis=1)
        fz = np.dot(hidden_layer_prediction,theta2.T)
        J = 0
        for r in range(0,m):
            J+= y[r]*np.log(sigmoid(fz[r])) + (1-y[r])*np.log(1 - sigmoid(fz[r]))
        J = -J + l*(np.sum(theta1*theta1) + np.sum(theta2*theta2) - np.size(theta1[:,0]) - np.size(theta2[:,0]))/(2*m)
        print("Iteration: ",k," Cost: ", J)
        [t_precision,t_recall,t_F1] = predict(theta1,theta2,X,y)
        [c_precision,c_recall,c_F1] = predict(theta1,theta2,X_CV,y)
        print("Train : ",[t_precision,t_recall,t_F1],"Cross Validate : ",[c_precision,c_recall,c_F1])

        #if k>20:
         #   g1,g2 = 0.9,0.9
            #rate = 0.01
        J_Arr = np.append(J_Arr,J)
        t_precision_Arr, t_recall_Arr, t_F1_Arr = np.append(t_precision_Arr,t_precision),\
                                                  np.append(t_recall_Arr,t_recall),np.append(t_F1_Arr,t_F1)
        c_precision_Arr, c_recall_Arr, c_F1_Arr = np.append(c_precision_Arr,c_precision),\
                                                  np.append(c_recall_Arr,c_recall),np.append(c_F1_Arr,c_F1)
        theta1_Arr = np.append(theta1_Arr,theta1.sum())
        theta2_Arr = np.append(theta2_Arr,theta2.sum())
        k+=1


print("Train : ",predict(theta1,theta2,X,y),"Cross Validate : ",predict(theta1,theta2,X_CV,y))
print ("time elapsed: ", format(time.time() - start_time))

#####################################################
[train_data,id] = load_data('kaggle_test_data')
train_data.drop(train_data.columns[[5,7,8,9,19,21,25,26,34,47,48,49,51,52,53,54,55,65,64,63]],axis =1,inplace=True)
train_data.to_csv('stuff.csv')
X_old_data = train_data.as_matrix()
m = len(X_old_data[:,1])
X_data = np.zeros(13*m).reshape(m,13)
X_data[:,0:7]=X_old_data[:,0:7]
X_data[:,12]=X_old_data[:,46]
for i in range(m):
    occMap = {7:-2,8:-6,9:-1,10:-3,11:-5,12:-4,13:6,14:3,15:2,16:1,17:4,18:5}
    for j in range(7,19):
        if(X_old_data[i,j]==1): X_data[i,7] = occMap[j]

    marstatMap = {19:1,20:-3,21:-1,22:-2,23:2}
    for j in range(19,24):
        if(X_old_data[i,j]==1): X_data[i,8] = marstatMap[j]

    eduMap = {24:-3,25:-3,26:-3,27:-3,28:-3,29:-4,
              30:-5,31:-1,32:-6,33:-1,34:3,35:5,36:4,37:2}
    for j in range(24,38):
        if(X_old_data[i,j]==1): X_data[i,9] = eduMap[j]

    relMap = {39:1,40:-2,41:-1,42:2,43:3}
    for j in range(39,44):
        if(X_old_data[i,j]==1): X_data[i,10] = relMap[j]

    partMap = {44:2,45:1}
    for j in range(44,46):
        if(X_old_data[i,j]==1): X_data[i,11] = partMap[j]

#np.savetxt('stuff2.csv',X_data,delimiter=',')

#Feature Normalization
[X0_mean , X0_variance, X_data[:,0]] = normalize(X_data[:,0])
[X1_mean , X1_variance, X_data[:,1]] = normalize(X_data[:,1])
[X1_mean , X1_variance, X_data[:,7]] = normalize(X_data[:,7])
[X1_mean , X1_variance, X_data[:,8]] = normalize(X_data[:,8])
[X1_mean , X1_variance, X_data[:,9]] = normalize(X_data[:,9])
[X1_mean , X1_variance, X_data[:,10]] = normalize(X_data[:,10])
X_data = np.append(np.ones(m).reshape(m,1),X_data,axis=1)

##########################################################################
m = len(X_data[:,1])
hidden_layer_prediction = sigmoid(np.dot(X_data,theta1.T))
hidden_layer_prediction = np.append(np.ones(m).reshape(m,1),hidden_layer_prediction,axis=1)
h2 = np.dot(hidden_layer_prediction,theta2.T)
h2 = sigmoid(h2)
#for i in range(len(h2)):
#    if h2[i]>0.5: h2[i]=1
#    else: h2[i] = 0

output('predictions.csv',h2,id)


figJ = plt.figure()
plt.plot(J_Arr)
#plt.plot(theta1_Arr)
#plt.plot(theta2_Arr)
#plt.legend(['5','3','1','0.3','0.1','0.03'], loc = 'upper right')
plt.title('Cost v/s Iterations')
plt.xlabel('Iteration ( x 10^3)')
plt.ylabel('Cost')
plt.show()
figJ.savefig('Cost.jpg')

figT = plt.figure()
plt.plot(t_precision_Arr)
plt.plot(t_recall_Arr)
plt.plot(t_F1_Arr)
plt.title('Train Score v/s Iterations')
plt.xlabel('Iteration ( x 10^3)')
plt.ylabel('Score')
plt.ylim((0,1))
plt.legend(['Precision','Recall','F1 score'],loc = 'upper left')
plt.show()
figT.savefig('Train.jpg')

figC = plt.figure()
plt.plot(c_precision_Arr)
plt.plot(c_recall_Arr)
plt.plot(c_F1_Arr)
plt.title('CV score v/s Iterations')
plt.xlabel('Iteration ( x 10^3)')
plt.ylabel('Score')
plt.ylim((0,1))
plt.legend(['Precision','Recall','F1 score'],loc = 'upper left')
plt.show()
figC.savefig('Test.jpg')

