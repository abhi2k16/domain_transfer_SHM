import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import scipy.stats as stats

############################### Load Exp data #################################
x_data_fe = pd.read_excel("D:/Vibration_Model_stiffplate/Domain_Adapt/FE_data_105_10_to_425_150_norm_-1_+1.xlsx",header = None) 
x_data_fe_np = x_data_fe.to_numpy() 
x_dam_fe_sym = x_data_fe_np[0:208,242] 
x_dam =[]# np.empty([416,1])
for i in range(20):
    file = 'D:/Vibration_Model_stiffplate/Domain_Adapt/New_Exp_2022/plate_dim_330x530_dam_at_265_050/Scan_'+str(i+2)+'_46_Hz_disp_abs.xlsx'
    x = pd.read_excel(file).to_numpy()
    x_d = x[10:426,4]
    x_dam.append(x_d)
x_dam_np = np.array(x_dam)
x_dam_np_sym1 = x_dam_np[:,0:208]
x_dam_np_sym2 = x_dam_np[:,208:416]
# Flip the sym2 data to make like sym1 data
x_dam_np_sym2 = np.flip(x_dam_np_sym2)
# Plot Data
for i in range(33):
    plt.plot(x_dam_np_sym2[i+2,:])
plt.show()

######################## Normalize Exp data ###################################
x_dam_np_sym1_norm = np.zeros([20,208])
for i in range(20):
    for j in range(208):
        n = (x_dam_np_sym1[i,j]-np.min(x_dam_np_sym1[i,:]))/(np.max(x_dam_np_sym1[i,:])-np.min(x_dam_np_sym1[i,:]))
        x_dam_np_sym1_norm[i,j] = n

x_dam_np_sym2_norm = np.zeros([20,208])        
for i in range(20):
    for j in range(208):
        n = (x_dam_np_sym2[i,j]-np.min(x_dam_np_sym2[i,:]))/(np.max(x_dam_np_sym2[i,:])-np.min(x_dam_np_sym2[i,:]))
        x_dam_np_sym2_norm[i,j] = n
# Concanate Sym1 and Sym2 norm data to make larger dataset
x_dam_np_sym_norm = np.concatenate((x_dam_np_sym1_norm,x_dam_np_sym2_norm), axis=0)
# Plot Normalizzed Data
for i in range(20):
    plt.plot(x_dam_np_sym_norm[i+2,:])
plt.show()       
# Pandas DataFrame
column = list(np.arange(1,209))
dam_x_df = pd.DataFrame(x_dam_np_sym2_norm,columns=column)

# Corelation Matrix
co_rel = dam_x_df.corr()
plt.imshow(co_rel)
plt.show()

########### bootstrapping of all column data in loop ##########################
dam_bst_data = []   # Empty List
for j in range(208):
    bst = pd.DataFrame({str(j+1):[dam_x_df.sample(17,replace=True)[j+1].mean() for i in range(0,200)]})
    dam_bst_data.append(bst)
# List to data frame in loop by adding column
dam_bst_df = pd.DataFrame() #Empty DataFrame
for i in range(208): 
    dam_bst_df[i+1] = dam_bst_data[i]
    
#plot bst data
for i in range(207):
    plt.plot(i+1,dam_bst_df[i+1][0],'r.')
    #print(bst_df[i+1][0])
plt.plot(x_dam_fe_sym,'g-')
plt.show()

# Corelationmatrix after bootstrap
bst_co_rel = dam_bst_df.corr()
plt.imshow(bst_co_rel)
plt.show()

################# Estimete mean and sd of original Exp Data ###################
dam_x_mean_df = []
for i in range(208):
    m = dam_x_df[i+1].mean()
    dam_x_mean_df.append(m)
    
dam_x_std_df = []
for i in range(208):
    sd = dam_x_df[i+1].std()
    dam_x_std_df.append(sd)

##################### estimete mean and sd of bst_df ##########################
dam_mean_df = []
for i in range(208):
    m = dam_bst_df[i+1].mean()
    dam_mean_df.append(m)
    
dam_std_df = []
for i in range(208):
    sd = dam_bst_df[i+1].std()
    dam_std_df.append(sd)
####################### Plot Histogram of data ################################
fig, ax = plt.subplots(5,5)
for i in range(5):
     for j in range(5):
         ax[i][j].hist(dam_bst_df[i*5+j+101],label=str(i*5+j+1))  
         ax[0, 1].set_title(str(i*5+j+1))

plt.show()  

# Plot Histogram of data using SeaBorn
fig, axes = plt.subplots(5,5, figsize=(16, 16))
for col, ax in zip(dam_x_df.columns, axes.flatten()):
    sns.histplot(dam_bst_df, x= col+1,kde = True, legend=ax==axes[0,0], ax=ax)
    #ax.set_title(col+51)
  
################### Generate Random Normal Dristribution data #################
import random
#norm_dist = [random.normalvariate(0.62,.09) for i in range(100)]
dam_norm_df_1000 = []
for i in range(208):
    norm_d = [random.gauss(dam_mean_df[i],dam_std_df[i]) for j in range(1000)]
    dam_norm_df_1000.append(norm_d)

################# Plot Generated data #########################################
# List to np array
dam_norm_df_np = np.array(dam_norm_df_1000)

# save array into csv file
np.savetxt("x_dam_265_050_46Hz_sym2_100.csv", dam_norm_df_np,
              delimiter = ",")

for i in range(1):
    plt.plot(dam_norm_df_np[:,i],'r-.')
plt.plot(x_dam_fe_sym,'b-')
plt.legend(['Exp','FE_Simulation'])
plt.title('50 mm Deb. at 265 mm')
plt.show()

##################### standard error ###########################
std_err_x = dam_x_df.sem()
std_err_bst = dam_bst_df.sem()
std_err_bst_200 = dam_bst_df.sem()
# plot std err
plt.plot(std_err_x,'r-')
plt.plot(std_err_bst,'b-')
plt.plot(std_err_bst_200,'g-')
plt.legend(['original sample','35 bootstrap sample','200 bootstrap sample'])
plt.xlabel('measurment points')
plt.ylabel('standard error')
plt.show()
    

