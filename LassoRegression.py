import GPyOpt
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import math
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
#import sys
from sklearn.metrics import mean_squared_error, r2_score
import os
#import shutil
from datetime import datetime
#import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

np.random.seed(42)
# current date and time

from math import floor, log10

def fexp(f):
    return int(floor(log10(abs(f)))) if f != 0 else 0

def fman(f):
    return f/10**fexp(f)

# Make Report Dir and Log-->
if not os.path.exists('Results'):
    os.makedirs('Results')    
    
#items = os.listdir(os.getcwd()+"\Results")
#matching = [s for s in items if "Report" in s]
report_count = 0

while True:
    if not os.path.exists(r'Results'+chr(92)+str(report_count)+'_Report'):
        os.makedirs(r'Results'+chr(92)+str(report_count)+'_Report')
        break
    else:
        report_count = report_count+1
        
save_location = r'Results'+chr(92)+str(report_count)+'_Report'+chr(92)
#combine dataset is the data set with EDX data
dataset_file = r"Combined-Data-Set.txt";
initial_dataset = pd.read_csv(dataset_file, sep='\t',header=0)
y = initial_dataset["Onset"].to_numpy()

#plot data is the reduced data set where the X and Y values are averaged over the 12 samples
plot_data_file = r"Plot-Data.txt";
plot_data = pd.read_csv(plot_data_file, sep='\t',header=0)
plot_data = plot_data[["HEA","Pt-EDX","Ru-EDX","Pd-EDX","Rh-EDX","Au-EDX","Average_Onset"]]
test_X = pd.DataFrame()
test_X["Pt"] = plot_data["Pt-EDX"]/100
test_X["Ru"] = plot_data["Ru-EDX"]/100
test_X["Pd"] = plot_data["Pd-EDX"]/100
test_X["Rh"] = plot_data["Rh-EDX"]/100
test_X["Au"] = plot_data["Au-EDX"]/100
test_y = plot_data["Average_Onset"].to_numpy()

#The amount of polynomial degrees
k = 2
#Include the intercept in the fitting
intercept = True

X = initial_dataset[["Pt","Ru","Pd","Rh","Au"]]/100
#X = initial_dataset[["Pt","Ru","Pd","Rh"]]/100
poly = PolynomialFeatures(degree=k, include_bias=False).fit(X)
columnsnames = poly.get_feature_names(X.columns) 
coefficients_list = ["alpha", "intercept"]

best_coefficients = pd.DataFrame(columns=(coefficients_list))

X_transformed = poly.fit_transform(X.to_numpy())
X_transformed = pd.DataFrame(X_transformed, columns=columnsnames)

#pieces of code to exclude lower orders of polynomial degrees in fitting
#poly2 = PolynomialFeatures(degree=k, include_bias=False).fit(X)
#columnsnames_reduced = poly2.get_feature_names(X.columns) 

#X_transformed = X_transformed.drop(columnsnames_reduced, axis=1)

for col in X_transformed.columns: 
    coefficients_list.append(col)

            
coefficients_list.append("RMSE")
coefficients_list.append("MAE")
coefficients_list.append("R^2")

X = X_transformed.to_numpy()

test_X = poly.fit_transform(test_X.to_numpy())
test_X = pd.DataFrame(test_X, columns=columnsnames).to_numpy()
#test_X = test_X.drop(columnsnames_reduced, axis=1).to_numpy()

#create an initial alpha guess
best_alpha = 0.1
old_alpha = 0
exp_best_alpha = fexp(best_alpha)
c = -1

#try out different alphas untill the change in best alpha becomes too small
while abs(old_alpha-best_alpha) > (10**(exp_best_alpha-2)):
    c = c+1         
    exp_best_alpha = fexp(best_alpha)
    dataset_size = X.shape[0]
    kfold = (X.shape[0])/(12)
    best_RMSE = 100
    print("iteration: " + str(c))
    print("Best Alpha: " + str(best_alpha))
    print("D_alpha: " + str(abs(old_alpha-best_alpha)))
    
    old_alpha = best_alpha
    
    alpha_list = (((np.arange(101))/100)*best_alpha*1.1)
    
    alpha_list = alpha_list[alpha_list > 0]

    #cross validation for different alphas
    for i in range(alpha_list.shape[0]):    
        RMSE = 0
        MAE = 0
        reg = linear_model.Lasso(alpha=alpha_list[i], fit_intercept=intercept)
        for n in range(int(math.ceil(kfold))):   
            idx = np.array(range(X.shape[0]))
            idx_test = np.logical_and(idx>=(((X.shape[0])/kfold*n)), idx<(((X.shape[0])/kfold*(n+1))))
            idx_train = np.logical_not(idx_test)
            
            reg = reg.fit(X[idx_train], y[idx_train])          
            RMSE += np.sum(np.square(reg.predict(test_X[n].reshape(1, -1))-test_y[n]))
            MAE += np.sum(abs(reg.predict(test_X[n].reshape(1, -1))-test_y[n]))  
        #68 is the size of the reduced data set
        RMSE /= 68
        MAE /= 68            
        RMSE = math.sqrt(RMSE)
        #print(str(k+1) + " nr " + str(i))   
        reg.fit(X,y)
        #print(reg.coef_)
    
        
        data = [alpha_list[i], reg.intercept_]
        for j in range(len(reg.coef_)):
            data.append(reg.coef_[j])
        data.append(RMSE)
        data.append(MAE)
        data.append(r2_score(y, reg.predict(X)))
       
        if(RMSE < best_RMSE):
            best_RMSE = RMSE
            best_alpha = alpha_list[i]
            best_coefficients = pd.DataFrame([data], columns=coefficients_list)
       
        header = False
        if i == 0:
            header = True
        pd.DataFrame([data], columns=coefficients_list).to_csv(save_location+'lasso_coefficients_'+str(k+1)+'_('+str(c)+').txt', sep='\t', index=False, header = header, mode="a")        
    
    #report the best alpha per loop with the associated Coefficients RMSE and MAE score
    best_coefficients = best_coefficients.loc[:, (best_coefficients != 0).any(axis=0)]
    best_coefficients.to_csv(save_location+'best_lasso_coefficients_'+str(k+1)+'_('+str(c)+').txt', sep='\t', index=False, mode="a")       
