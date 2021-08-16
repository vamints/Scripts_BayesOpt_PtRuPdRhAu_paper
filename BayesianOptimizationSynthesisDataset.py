import GPyOpt
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import math
#import sys
import os
#import shutil
from datetime import datetime
#import random
import matplotlib.pyplot as plt

np.random.seed(42)
# current date and time

#parameters for dataset location and reports-->
results_HEA_list = ["ID","Iteration","Pt","Ru","Pd","Rh","Au","Onset"]
results_Hyper_list = ["Iteration","n_estimators","Bootstrap","Max_depth","Max_features","Min_samples_leaf","Min_samples_split","criterion","RMSE_test","MAE_test","RMSE_train","MAE_train","k error","Error"]
results_HEA = pd.DataFrame(columns=(results_HEA_list))
results_Hyper = pd.DataFrame(columns=(results_Hyper_list))
iteration_HEA = 0
iteration_Hyper = 0
dataset_file = r"Data_Set_Shuffled.txt";
initial_dataset = pd.read_csv(dataset_file, sep='\t',header=0)
X = initial_dataset[["Pt","Ru","Pd","Rh","Au"]].to_numpy()
y = initial_dataset["Onset"].to_numpy()



# --- Progress Trackers and Max iteration size. If 0 only the input X parameters will be investigated
max_iteration_HEA = 50
max_iteration_Hyper = 700
last_report_progress = 0
progress_step_report = 5

print("--------------------------------------Start HEA Machine Learning Program")



#Hyperparameter tuning settings
#size of test batches 
ksize = 5
#first points in the data set can be excluded from the k-fold crossfalidation. 0 to ignor
corner_points = 0
#condition on RMSE (true) or MAE (false)
error_condition = True



#Use model values for Bayesian optimization (true) or use experimental averages (false)
use_model = True
#Samples selected with the Bayesian optimization can be filtered based on their distance to known samples. Set the value to the desired distance
data_distance = 10

# --- Hyperparameter Tuning

#Set start parameters for the Random Forrest Regressor Hyperparameter tuning
#Set max_iterations_Hyper to 0 if an optimum is already found 5440
n_estimators = 515           
bootstrap = True
max_depth = 172
max_features = 'auto'
min_samples_leaf = 1
min_samples_split = 2
criterion = "mae"

#Encoding the parameters into an array
code_max_features = 0

if max_features == 'sqrt':
    code_max_features = 1
elif max_features == 'log2':
    code_max_features = 2
else:
    code_max_features = 0
    
if max_depth is None:
   max_depth = 300
   
code_criterion = 0
if criterion == "mse":
       code_criterion = 1


start_config_Hyper = [[n_estimators, int(bootstrap), max_depth, code_max_features, min_samples_leaf, min_samples_split, code_criterion]]

# Make Report Dir and Log-->
if not os.path.exists('Results'):
    os.makedirs('Results')    
    
#items = os.listdir(os.getcwd()+"\Results")
#matching = [s for s in items if "Report" in s]
report_count = 0

#Create new results folder
while True:
    if not os.path.exists(r'Results'+chr(92)+str(report_count)+'_Report'):
        os.makedirs(r'Results'+chr(92)+str(report_count)+'_Report')
        break
    else:
        report_count = report_count+1
        
save_location = r'Results'+chr(92)+str(report_count)+'_Report'+chr(92)

#Make backup of the used Data-set for reproducibility
initial_dataset.to_csv(save_location+'Data_Set_Shuffled.txt', sep='\t', index=False)  
 
#Make log to save progress in case of program crashes or is manually stopped
file = open(save_location + 'log1.txt', "a")
file.write('------------------------------ Hyper Log ---------------------------------\n')
file.close()

print("--------------------------------------Saving log at "+ save_location)

print("--------------------------------------Start Hyper Parameter Optimization at: "+str(datetime.now().strftime("%H:%M:%S")))

#Define the hyperparameter tuning. Post research this loop might be redundant and default settings can be used
def f_hyp(x):
    global X,y,ksize,criterion,progress_step_report,error_condition,corner_points,bootstrap,max_depth,max_features,min_samples_leaf,min_samples_split,n_estimators,results_Hyper_list,iteration_Hyper,results_Hyper,last_report_progress,save_location
    n_estimators = int(x[0,0])
    bootstrap = bool(x[0,1])
    iteration_Hyper = iteration_Hyper+1
    if x[0,2] < 300:
        max_depth = int(x[0,2])
    else:
        max_depth = None
    
    if x[0,3] == 1:
        max_features = 'sqrt'
    elif x[0,3] == 2:
        max_features = 'log2'
    else:
        max_features = 'auto'
        
    min_samples_leaf = int(x[0,4])
    min_samples_split = int(x[0,5])
    
    if bool(x[0,6]):
        criterion = 'mse'
    else:
        criterion = 'mae'
        
    #the data set used in this research had 3 samples grouped together. Modify the 3 for bigger or smaller groups
    kfold = (X.shape[0]-corner_points*3)/(3*ksize)
    error = 0
    RMSE = 0
    MAE = 0   
    k_error = np.empty([0, 0])
    
    #kfold cross validation loop
    for n in range(int(math.ceil(kfold))):   
        idx = np.array(range(X.shape[0]))
        idx_test = np.logical_and(idx>=(((X.shape[0]-corner_points)/kfold*n)+corner_points), idx<(((X.shape[0]-corner_points)/kfold*(n+1))+corner_points))
        idx_train = np.logical_not(idx_test)
        clf = RandomForestRegressor(n_estimators=n_estimators, 
                                    bootstrap=bootstrap, 
                                    max_depth=max_depth, 
                                    max_features=max_features, 
                                    min_samples_leaf=min_samples_leaf,
                                    min_samples_split=min_samples_split,
                                    criterion=criterion,
                                    random_state=42,
                                    n_jobs=-1)        
        clf = clf.fit(X[idx_train], y[idx_train])     
        
        RMSE += np.sum(np.square(clf.predict(X[idx_test])-y[idx_test]))
        MAE += np.sum(abs(clf.predict(X[idx_test])-y[idx_test]))        
        k_error = np.append(k_error, (clf.predict(X[idx_test])-y[idx_test]))
    #Report kfold results
    RMSE /= X.shape[0]
    MAE /= X.shape[0]       
    RMSE = math.sqrt(RMSE)     
    clf = RandomForestRegressor(n_estimators=n_estimators, 
                                bootstrap=bootstrap, 
                                max_depth=max_depth, 
                                max_features=max_features, 
                                min_samples_leaf=min_samples_leaf,
                                min_samples_split=min_samples_split,
                                criterion=criterion,
                                random_state=42,
                                n_jobs=-1)
    clf = clf.fit(X, y)     
    train_RMSE = math.sqrt(np.square(clf.predict(X)-y).mean())
    train_MAE = abs(clf.predict(X)-y).mean()
    
    
    if error_condition:
        error = RMSE
    else:
        error = MAE
        
    results_Hyper = results_Hyper.append(pd.DataFrame([[iteration_Hyper,n_estimators,bootstrap,max_depth,max_features,min_samples_leaf,min_samples_split,criterion,RMSE,MAE,train_RMSE,train_MAE,k_error,error]], columns=results_Hyper_list))
    
    log = pd.DataFrame([[iteration_Hyper,n_estimators,bootstrap,max_depth,max_features,min_samples_leaf,min_samples_split,criterion,RMSE,MAE,train_RMSE,train_MAE,k_error,error]], columns=results_Hyper_list)
    header = False
    if iteration_Hyper == 1:
        header = True
    log.to_csv(save_location+'log1.txt', sep='\t', index=False, header=header, mode="a")
    
    temp = math.floor((iteration_Hyper)/(max_iteration_Hyper+1)*100/progress_step_report)*progress_step_report
    if last_report_progress < temp :
        print("Progress: " + str(temp)+"%")
    last_report_progress = temp
    
    return error   

#Define the domains of the hyperparameter values    
domainspan_hyper_estimator = (np.arange(2500)+1)*5
domainspan_hyper_depth = np.arange(300)+1
domainspan_hyper_leaf = np.arange(10)+1
domainspan_hyper_split = (np.arange(10)+1)*2

"""
domain_hyper = [{'name': 'n_estimators', 'type': 'discrete', 'domain': [20, 40, 80, 100, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2500, 3000, 4000, 5000, 10000, 15000]},
          {'name': 'bootstrap', 'type': 'discrete', 'domain': [0,1]},
          {'name': 'max_depth', 'type': 'discrete', 'domain': [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200, 250]},
          {'name': 'max_features', 'type': 'discrete', 'domain': [0,1,2]},
          {'name': 'min_samples_leaf', 'type': 'discrete', 'domain': [1, 2, 4, 8, 12, 16, 20, 24, 32]},
          {'name': 'min_samples_split', 'type': 'discrete', 'domain': [2, 4, 6, 8, 12, 16, 20, 24, 28, 32]},
          {'name': 'criterion', 'type': 'discrete', 'domain': [0,1]}]
"""
domain_hyper = [{'name': 'n_estimators', 'type': 'discrete', 'domain': domainspan_hyper_estimator},
          {'name': 'bootstrap', 'type': 'discrete', 'domain': [0,1]},
          {'name': 'max_depth', 'type': 'discrete', 'domain': domainspan_hyper_depth},
          {'name': 'max_features', 'type': 'discrete', 'domain': [0,1,2]},
          {'name': 'min_samples_leaf', 'type': 'discrete', 'domain': domainspan_hyper_leaf},
          {'name': 'min_samples_split', 'type': 'discrete', 'domain': domainspan_hyper_split},
          {'name': 'criterion', 'type': 'discrete', 'domain': [0,1]}]

default_values_X = np.array(start_config_Hyper)
#default_values_Y = np.zeros([1, 1])
#default_values_Y[0] = f_hyp(default_values_X)

#Bayesian optimization of the RandomForest regression
HyperOpt = GPyOpt.methods.BayesianOptimization(f=f_hyp, domain=domain_hyper,
                                               X = default_values_X,
                                               batch_size=1, de_duplication=True)
HyperOpt.run_optimization(max_iter=max_iteration_Hyper)

print("--------------------------------------Finished Hyper Parameter Optimization at: "+str(datetime.now().strftime("%H:%M:%S")))
### Save Report



# --- Set Parameters for model
hyper_best = HyperOpt.X[np.argmin(HyperOpt.Y)]
n_estimators = int(hyper_best[0])
bootstrap = bool(hyper_best[1])
if hyper_best[2] < 300:
    max_depth = int(hyper_best[2])
else:
    max_depth = None    
if hyper_best[3] == 1:
        max_features = 'sqrt'
elif hyper_best[3] == 2:
    max_features = 'log2'
else:
    max_features = 'auto'
min_samples_leaf = int(hyper_best[4])
min_samples_split = int(hyper_best[5])
if bool(hyper_best[6]):
        criterion = 'mse'
else:
    criterion = 'mae'
        

print("--------------------------------------Saving log at "+ save_location)
 
print("--------------------------------------Start HEA Optimization at: "+str(datetime.now().strftime("%H:%M:%S")))

file = open(save_location + 'log2.txt', "a")
file.write('------------------------------ HEA Optimization Log ---------------------------------\n')
file.close()

last_report_progress = 0

#create initial points for the bayesian optimization. In this model is assumed that X values are constant and Y values are repeats
initial_X = initial_dataset[["Pt","Ru","Pd","Rh"]].drop_duplicates(subset=None, keep='first', inplace=False).to_numpy()
initial_points = len(initial_X);
initial_Y = np.zeros([initial_points, 1])

#Make best model Random Regression model of HEA
clf = RandomForestRegressor(n_estimators=n_estimators, 
                                    bootstrap=bootstrap, 
                                    max_depth=max_depth, 
                                    max_features=max_features, 
                                    min_samples_leaf=min_samples_leaf,
                                    min_samples_split=min_samples_split,
                                    criterion=criterion,
                                    random_state=42)
clf = clf.fit(X, y) 
    
HEA_id_list = initial_dataset["HEA"].drop_duplicates(keep='first', inplace=False)
HEA_id_list = HEA_id_list.reset_index().drop(columns=['index'])

# --- Bayesian optimization problem Problem
def f(x):
    global HEA_id_list,use_model,progress_step_report,results_HEA,iteration_HEA,results_HEA_list,initial_dataset,clf,initial_points,max_iteration_HEA,last_report_progress,save_location
    iteration_HEA = iteration_HEA+1      
    x = np.append(x,[100-x[:,0]-x[:,1]-x[:,2]-x[:,3]])
    x = np.reshape(x,(1,-1))
    
    HEA_id = None
    
    predicted = clf.predict(x)
    #print(r"Iteration: " + str(iteration) + "\nPoint: "+str(x)+" Prediction: "+str(predicted))
    #count if sample is part of initial and already investigated samples or a new sample
    iteration_str = iteration_HEA-initial_points   
    if iteration_HEA <= initial_points:
        iteration_str = None  
        temp = iteration_HEA+1
        HEA_id = HEA_id_list["HEA"][iteration_HEA-1]
    
    #save results of optimization
    results_HEA = results_HEA.append(pd.DataFrame([[HEA_id, iteration_str, x[:,0][0], x[:,1][0], x[:,2][0], x[:,3][0], x[:,4][0], predicted[0]]], columns=results_HEA_list))
    
    #Save log file in case program crashes or is manually stopped not to lose experiments
    log = pd.DataFrame([[HEA_id, iteration_str, x[:,0][0], x[:,1][0], x[:,2][0], x[:,3][0], x[:,4][0], predicted[0]]], columns=results_HEA_list)
    header = False
    if iteration_str == 1:
        header = True
    log.to_csv(save_location+'log2.txt', sep='\t', index=False, header=header, mode="a")
    
    #Track progress
    temp = 0
    if use_model :
        temp = math.floor((iteration_HEA)/(max_iteration_HEA+initial_points)*100/progress_step_report)*progress_step_report
    else :
        temp = math.floor((iteration_HEA-initial_points)/(max_iteration_HEA)*100/progress_step_report)*progress_step_report
    if last_report_progress < temp :
        print("Progress: " + str(temp)+"%")
    last_report_progress = temp   
    return predicted

#input parameters for the optimization -->
domainspan = np.arange(96)+1

domain = [{'name': 'var_1', 'type': 'discrete', 'domain': domainspan},
          {'name': 'var_2', 'type': 'discrete', 'domain': domainspan},
          {'name': 'var_3', 'type': 'discrete', 'domain': domainspan},
          {'name': 'var_4', 'type': 'discrete', 'domain': domainspan}]

#The expression corresponding to 'constraint' is an inequality that is less than zero.
constraints = [{'name':'const_1', 'constraint': 'x[:,0]+x[:,1]+x[:,2]+x[:,3]+1-100'}]

#The initial points that are used in the bayesian optimization can be fed either using experimental data or predicted data
#When the data is fed using the initial data an average is calculated for each group of 3 samples. This loop needs to be modified when the sample size changes
if use_model :
    i = 0
    for i in range (initial_points):
        initial_Y[i] = f(np.reshape(initial_X[i],(1,-1)))
else :
     i = 0
     for i in range (initial_points):
        iteration_HEA = iteration_HEA+1
        initial_Y[i] = initial_dataset["Onset"][i*3]*1/3+initial_dataset["Onset"][i*3+1]*1/3+initial_dataset["Onset"][i*3+2]*1/3
        results_HEA = results_HEA.append(pd.DataFrame([[i+1, None, initial_dataset["Pt"][i*3], initial_dataset["Ru"][i*3], initial_dataset["Pd"][i*3], initial_dataset["Rh"][i*3], initial_dataset["Au"][i*3], initial_Y[i][0]]], columns=results_HEA_list))
   
#perform the bayesian optimziation of the samples
HEAOpt = GPyOpt.methods.BayesianOptimization(f=f, domain=domain,
                                             initial_design_numdata=initial_points,
                                             X=initial_X,
                                             Y=initial_Y,
                                             acquisition_type="EI", 
                                             constraints=constraints, 
                                             batch_size=1, 
                                             de_duplication=True,
                                             maximize=False)
HEAOpt.run_optimization(max_iter=max_iteration_HEA)

#Make report -->

#Report the results of the hyperparameter tuning.
results_Hyper = results_Hyper.sort_values(['Error'],ascending=False)
results_Hyper = results_Hyper.drop(columns=['Error'])

HyperOpt.plot_convergence(save_location + 'Hyper_convergence.png')
HyperOpt.save_report(save_location + 'Hyper_report.txt')
file = open(save_location + 'Hyper_report.txt', "a")

file.write('\n\n------------------------------ Hyperparameter tuning configuration ---------------------------------\n')
file.write('k-size = ' + str(ksize)+'\n')
if error_condition:
    file.write('Error = RMSE')
else:
    file.write('Error = MAE')
file.write('\n\n------------------------------ Iteration Results ---------------------------------\n')
file.close()
results_Hyper.to_csv(save_location + 'Hyper_report.txt', sep='\t', index=False, mode="a")

#Create Bayesian optimization report

HEAOpt.save_report(save_location+'HEA_report.txt')

HEAOpt.plot_convergence(save_location+'HEA_convergence.png')


results_HEA['Known'] = results_HEA['Iteration'].isnull()
results_HEA['Interest'] = (results_HEA['Iteration'] > 0)  
results_HEA['Distance to Known'] = 10000.0
results_HEA['Closest HEA ID'] = None

results_HEA = results_HEA.reset_index().drop(columns=['index'])

for index, row in results_HEA.iterrows():
    if results_HEA['Known'][index]:
        results_HEA['Distance to Known'][index] = None
        for index_2, row_2 in results_HEA.iterrows():
            if not results_HEA['Known'][index_2]:
                distance = math.sqrt((row['Pt']-row_2['Pt'])**2+(row['Pd']-row_2['Pd'])**2+(row['Rh']-row_2['Rh'])**2+(row['Ru']-row_2['Ru'])**2+(row['Au']-row_2['Au'])**2) 
                if distance < results_HEA['Distance to Known'][index_2] :                    
                    results_HEA['Distance to Known'][index_2] = distance
                    results_HEA['Closest HEA ID'][index_2] = results_HEA['ID'][index]
                if distance < data_distance :                    
                    results_HEA['Interest'][index_2] = False

Temp = results_HEA.drop(columns=['Known', 'Interest'])

file = open(save_location+'HEA_report.txt', "a")
file.write('\n\n------------------------------ Iteration Results ---------------------------------\n')
file.close()

Temp.to_csv(save_location+'HEA_report.txt', sep='\t', index=False, mode="a")

results_HEA['Keep'] = results_HEA['Iteration'].isnull()                 

results_HEA = results_HEA.sort_values(['Onset'],ascending=True)

results_HEA = results_HEA.reset_index().drop(columns=['index'])

for index, row in results_HEA.iterrows():
    if results_HEA['Interest'][index]:
        results_HEA['Keep'][index] = True
        for index_2, row_2 in results_HEA.iterrows():
            if index < index_2:
                distance = math.sqrt((row['Pt']-row_2['Pt'])**2+(row['Pd']-row_2['Pd'])**2+(row['Rh']-row_2['Rh'])**2+(row['Ru']-row_2['Ru'])**2+(row['Au']-row_2['Au'])**2) 
                if distance < data_distance :
                    results_HEA['Interest'][index_2] = False

results_HEA = results_HEA[results_HEA.Keep != False]


results_HEA = results_HEA.drop(columns=['Known'])
results_HEA = results_HEA.drop(columns=['Interest'])
results_HEA = results_HEA.drop(columns=['Keep'])

results_HEA = results_HEA.sort_values(['Onset'],ascending=False)

file = open(save_location+'HEA_report.txt',"a")
file.write('\n\n------------------------------ Selection ---------------------------------\n')
file.close()

results_HEA.to_csv(save_location+'HEA_report.txt', sep='\t', index=False, mode="a")

print("--------------------------------------Finished HEA Optimization at: "+str(datetime.now().strftime("%H:%M:%S")))
print("--------------------------------------Saved Files at "+ save_location)

#sys.exit()
