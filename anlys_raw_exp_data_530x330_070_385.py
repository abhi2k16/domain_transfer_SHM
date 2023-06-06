import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

############################### Load Exp data #################################
x_data_fe = pd.read_excel("D:/Vibration_Model_stiffplate/Domain_Adapt/FE_data_105_10_to_425_150_norm_-1_+1.xlsx",header = None) 
x_data_fe_np = x_data_fe.to_numpy() 
x_dam_fe_sym = x_data_fe_np[0:208,242] 
x_dam_385_070_46Hz =[]# np.empty([416,1])
n_exp = 20 # number of of exp
for i in range(n_exp):
    file = r'D:/Vibration_Model_stiffplate/abhijeet_ldv_lab/New_exp_2022/plate_dim_330x530_dam_at_385_070/Scan_'+str(i+1)+'_48_Hz_disp_abs.xlsx'
    x = pd.read_excel(file).to_numpy()
    x_d = x[10:426,4]
    x_dam_385_070_46Hz.append(x_d)
x_dam_385_070_46Hz = np.array(x_dam_385_070_46Hz)
x_dam_385_070_46Hz_sym1 = x_dam_385_070_46Hz[:,0:208]
x_dam_385_070_46Hz_sym2 = x_dam_385_070_46Hz[:,208:416]
# Flip the sym2 data to make like sym1 data
x_dam_385_070_46Hz_sym2 = np.flip(x_dam_385_070_46Hz_sym2)
# Plot Data
for i in range(n_exp):
    if x_dam_385_070_46Hz_sym2[i,13] > 0.00020:
        print(i)
        plt.plot(x_dam_385_070_46Hz_sym2[i,:])      
plt.show()

for i in range(n_exp):
    if x_dam_385_070_46Hz_sym1[i,13] > 0.00020:
        print(i)
        plt.plot(x_dam_385_070_46Hz_sym1[i,:])      
plt.show()

######################## Normalize Exp data ###################################
x_dam_385_070_46Hz_sym1_norm = np.zeros([n_exp,208])
for i in range(n_exp):
    for j in range(208):
        n = (
            x_dam_385_070_46Hz_sym1[i,j]-np.min(x_dam_385_070_46Hz_sym1[i,:])
            )/(np.max(x_dam_385_070_46Hz_sym1[i,:])-np.min(x_dam_385_070_46Hz_sym1[i,:]))
        x_dam_385_070_46Hz_sym1_norm[i,j] = n

x_dam_385_070_46Hz_sym2_norm = np.zeros([8,208])        
for i in range(8):
    for j in range(208):
        n = (
            x_dam_385_070_46Hz_sym2[i,j]-np.min(x_dam_385_070_46Hz_sym2[i,:])
            )/(np.max(x_dam_385_070_46Hz_sym2[i,:])-np.min(x_dam_385_070_46Hz_sym2[i,:]))
        #n = (x_dam_385_070_46Hz_sym2[i,j]/np.max(np.max(x_dam_385_070_46Hz_sym2)))
        x_dam_385_070_46Hz_sym2_norm[i,j] = n
        
# Plot norm data
for i in range(8):
    plt.plot(x_dam_385_070_46Hz_sym2_norm[i,:])
plt.show()
# Pandas DataFrame
column = list(np.arange(1,209))
x_dam_385_070_46Hz_sym2_norm_df = pd.DataFrame(x_dam_385_070_46Hz_sym2_norm,columns=column)

########### bootstrapping of all column data in loop ##########################
x_dam_385_070_46Hz_sym2_bst = []   # Empty List
for j in range(208):
    bst = pd.DataFrame({str(j+1):[x_dam_385_070_46Hz_sym2_norm_df.sample(5,replace=True)[j+1].mean() for i in range(0,100)]})
    x_dam_385_070_46Hz_sym2_bst.append(bst)
# List to data frame in loop by adding column
x_dam_385_070_46Hz_sym2_bst_df = pd.DataFrame() #Empty DataFrame
for i in range(208): 
    x_dam_385_070_46Hz_sym2_bst_df[i+1] = x_dam_385_070_46Hz_sym2_bst[i]

##################### estimete mean and sd of bst_df ##########################
x_dam_385_070_46Hz_sym2_bst_df_mean = []
for i in range(208):
    m = x_dam_385_070_46Hz_sym2_bst_df[i+1].mean()
    x_dam_385_070_46Hz_sym2_bst_df_mean.append(m)
    
x_dam_385_070_46Hz_sym2_bst_df_sd = []
for i in range(208):
    sd = x_dam_385_070_46Hz_sym2_bst_df[i+1].std()
    x_dam_385_070_46Hz_sym2_bst_df_sd.append(sd)    
    
################### Generate Random Normal Dristribution data #################
import random
#norm_dist = [random.normalvariate(0.62,.09) for i in range(100)]
x_dam_385_070_46Hz_sym2_100 = []
for i in range(208):
    norm_d = [random.gauss(x_dam_385_070_46Hz_sym2_bst_df_mean[i],x_dam_385_070_46Hz_sym2_bst_df_sd[i]) for j in range(100)]
    x_dam_385_070_46Hz_sym2_100.append(norm_d)

################# Plot Generated data #########################################
# List to np array
x_dam_385_070_46Hz_sym2_100 = np.array(x_dam_385_070_46Hz_sym2_100)
"""
# save array into csv file
np.savetxt("x_dam_385_070_46Hz_sym2_100.csv", x_dam_385_070_46Hz_sym2_100,
              delimiter = ",")

for i in range(10):
    plt.plot(x_dam_385_070_46Hz_sym2_100[:,i])
#plt.plot(x_dam_fe_sym,'b-')
#plt.legend(['Exp','FE_Simulation'])
plt.title('70 mm Deb. at 385 mm')
plt.show()
"""