import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import scipy.stats as stats

############################### Load Exp data #################################
x_fe = pd.read_excel("D:/Vibration_Model_stiffplate/Domain_Adapt/FE_data_105_10_to_425_150.xlsx",header = None) 
x_fe_np = x_fe.to_numpy() 
x_fe_sym_np = x_fe_np[0:208,:]
# normalize(0-1) data
x_fe_sym_norm = (x_fe_sym_np - np.min(np.min(x_fe_sym_np)))/(np.max(np.max(x_fe_sym_np))-np.min(np.min(x_fe_sym_np))) 
# Save as Matlab file
"""
from scipy.io import savemat
a = np.arange(20)
mdic = {"data": x_fe_sym_norm, "label": "Fe_data_sym"}
savemat("Fe_data_sym.mat", mdic)
"""
# feature index with higher Mahalanobis Distances
idx1 = np.array(range(3,15))
idx2 = np.array(range(15,25))
idx3 = np.array(range(30,37))
idx4 = np.array(range(42,51))
idx5 = np.array(range(55,65))
idx6 = np.array(range(67,77))
idx7 = np.array(range(91,102))
idx8 = np.array(range(116,129))
idx9 = np.array(range(142,154))
idx10 = np.array(range(168,182))

idx_msd = []
for i in range(10):
    file = 'idx'+str(i+1)
    file = eval(file)  # eval is a method to convert str to object
    idx_msd.extend(file)
###############################################################################
x_fe_sym_norm_msd = np.zeros([108,494])
for i,j in enumerate(idx_msd):
    n = x_fe_sym_norm[j,:]
    x_fe_sym_norm_msd [i,:] = n

x_fe_sym_norm = x_fe_sym_norm_msd

###############################################################################
x_undam_fe_sym = x_fe_sym_norm[0:108,232]
x_undam =[]# np.empty([416,1])
for i in range(35):
    file = 'D:/Vibration_Model_stiffplate/Domain_Adapt/New_Exp_2022/plate_dim_330x530_dam_at_265_000/Scan_'+str(i+1)+'_46_Hz_disp_abs.xlsx'
    x = pd.read_excel(file).to_numpy()
    x_d = x[10:426,4]
    x_undam.append(x_d)
x_undam_np = np.array(x_undam)
x_undam_np_sym1 = x_undam_np[:,0:208]
# pull out the value coresponding to max MSD
x_undam_np_sym1_msd = np.zeros([35,len(idx_msd)])
for i,j in enumerate(idx_msd):
    n = x_undam_np_sym1[:,j]
    x_undam_np_sym1_msd[:,i] = n
x_undam_np_sym1 = x_undam_np_sym1_msd
x_undam_np_sym2 = x_undam_np[:,208:416]
# Flip the sym2 data to make like sym1 data
x_undam_np_sym2 = np.flip(x_undam_np_sym2)
# pull out the value coresponding to max MSD
x_undam_np_sym2_msd = np.zeros([35,len(idx_msd)])
for i,j in enumerate(idx_msd):
    n = x_undam_np_sym2[:,j]
    x_undam_np_sym2_msd[:,i] = n

x_undam_np_sym2 = x_undam_np_sym2_msd   
# Plot Data
for i in range(33):
    plt.plot(x_undam_np_sym2[i+2,:])
plt.show()

######################## Normalize Exp data ###################################
x_undam_np_sym1_norm = np.zeros([35,108])
for i in range(35):
    for j in range(108):
        n = (x_undam_np_sym1[i,j]-np.min(x_undam_np_sym1[i,:]))/(np.max(x_undam_np_sym1[i,:])-np.min(x_undam_np_sym1[i,:]))
        x_undam_np_sym1_norm[i,j] = n

x_undam_np_sym2_norm = np.zeros([35,108])        
for i in range(35):
    for j in range(108):
        n = (x_undam_np_sym2[i,j]-np.min(x_undam_np_sym2[i,:]))/(np.max(x_undam_np_sym2[i,:])-np.min(x_undam_np_sym2[i,:]))
        x_undam_np_sym2_norm[i,j] = n
# Concanate Sym1 and Sym2 norm data to make larger dataset
x_undam_np_sym_norm = np.concatenate((x_undam_np_sym1_norm,x_undam_np_sym2_norm), axis=0)
# Plot Normalizzed Data
for i in range(67):
    plt.plot(x_undam_np_sym_norm[i+2,:])
plt.show()       
# Pandas DataFrame
column = list(np.arange(1,109))
undam_x_df = pd.DataFrame(x_undam_np_sym2_norm,columns=column)

# Corelation Matrix
co_rel = undam_x_df.corr()
plt.imshow(co_rel)
plt.show()

########### bootstrapping of all column data in loop ##########################
undam_bst_data = []   # Empty List
for j in range(108):
    bst = pd.DataFrame({str(j+1):[undam_x_df.sample(20,replace=True)[j+1].mean() for i in range(0,200)]})
    undam_bst_data.append(bst)

## bootsraping using withought any parameter ####    
undam_bst_nonparmtr = []   # Empty List
for j in range(108):
    bst = [undam_x_df.sample(35,replace=True)[j+1] for i in range(0,100)]
    bst_np = np.array(bst).reshape([35*100,])
    undam_bst_nonparmtr.append(bst_np)

# List to dataframe of nonparametric bootstrap samples 
undam_bst_nonparmtr_df = pd.DataFrame()
for i in range(108):
    undam_bst_nonparmtr_df[i+1] = undam_bst_nonparmtr[i]

# List to data frame in loop by adding column
undam_bst_df = pd.DataFrame() #Empty DataFrame
for i in range(108): 
    undam_bst_df[i+1] = undam_bst_data[i]

########################## Check data is normal Dist #########################
from scipy.stats import zscore
import random
## Q-Q plot using scipy probplot function
zscore_1= zscore(undam_bst_df[13])
stats.probplot(zscore_1, dist="norm", plot=plt)
plt.show()  
# Q-Q plot using by obtainig z-score of data and theoretical standard data generated w/ 0 mean and 1 SD
z_data = sorted(np.array([random.gauss(0,1) for j in range(200)]))
z_score = np.array(zscore(undam_bst_df[1]))

fig, ax = plt.subplots(5,5, figsize = (12,12))
for i in range(5):
     for j in range(5):
         ax[i][j].plot(z_data, sorted(np.array(zscore(undam_bst_df[i*5+j+176]))),'b.',label=str(i*5+j+1))  
         ax[i][j].plot(z_data,z_data,'r-')
plt.show() 

# Corelationmatrix after bootstrap
bst_co_rel = undam_bst_df.corr()
plt.imshow(bst_co_rel)
plt.show()

################# Estimete mean and sd of original Exp Data ###################
undam_x_mean_df = []
for i in range(108):
    m = undam_x_df[i+1].mean()
    undam_x_mean_df.append(m)
    
undam_x_std_df = []
for i in range(108):
    sd = undam_x_df[i+1].std()
    undam_x_std_df.append(sd)

##################### estimete mean and sd of bst_df ##########################
undam_mean_df = []
for i in range(108):
    m = undam_bst_df[i+1].mean()
    undam_mean_df.append(m)
    
undam_std_df = []
for i in range(108):
    sd = undam_bst_df[i+1].std()
    undam_std_df.append(sd)
####################### Plot Histogram of data ################################
fig, ax = plt.subplots(5,5)
for i in range(5):
     for j in range(5):
         ax[i][j].hist(undam_bst_df[i*5+j+101],label=str(i*5+j+1))  
         ax[0, 1].set_title(str(i*5+j+1))
plt.show()  

# Plot Histogram of data using SeaBorn
fig, axes = plt.subplots(5,5, figsize=(16, 16))
for col, ax in zip(undam_x_df.columns, axes.flatten()):
    sns.histplot(undam_bst_df, x= col+1,kde = True, legend=ax==axes[0,0], ax=ax)
    #ax.set_title(col+51)
  
################### Generate Random Normal Dristribution data #################
import random
undam_norm_df_1000 = []
for i in range(108):
    norm_d = [random.gauss(undam_mean_df[i],undam_std_df[i]) for j in range(10000)]
    undam_norm_df_1000.append(norm_d)

################# Plot Generated data #########################################
# List to np array
undam_norm_df_np_1000 = np.array(undam_norm_df_1000)
for i in range(50):
    plt.plot(undam_norm_df_np_1000[:,i])
plt.plot(x_undam_fe_sym,'r-')
plt.show()

##################### standard error ###########################
std_err_x = undam_x_df.sem()
std_err_bst = undam_bst_df.sem()
std_err_bst_200 = undam_bst_df.sem()
# plot std err
plt.plot(std_err_x,'r-')
plt.plot(std_err_bst,'b-')
plt.plot(std_err_bst_200,'g-')
plt.legend(['original sample','35 bootstrap sample','200 bootstrap sample'])
plt.xlabel('measurment points')
plt.ylabel('standard error')
plt.show()
##################### ADD Noise to undam FE data ##############################
# Function to add % noise
def add_noise(noise_percent, x):
    r = np.random.normal(0,1,size=(x.shape))
    x_noise = x*(1+(noise_percent/100)*r)
    return x_noise
    
# generate the noisy data
undam_fe_n_data = np.zeros([108,10000])
for i in range(10000):
    n_data = add_noise(noise_percent = 0.5,x=x_undam_fe_sym)
    undam_fe_n_data[:,i] = n_data
    
# Plot noisy data 
for i in range(100):
    plt.plot(undam_fe_n_data[:,i])
    #plt.plot(undam_norm_df_np_1000[:,i])
plt.show()

###############################################################################
######################## DNN Model for domain Adaptation ######################
###############################################################################
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from torch import optim
import copy

device='cuda' if torch.cuda.is_available() else 'cpu'
print(device)
# Load Data
x_fe_tensor = torch.from_numpy(np.transpose(undam_fe_n_data)).reshape(10000,1,108).float().to(device)
x_exp_tensor = torch.from_numpy(np.transpose(undam_norm_df_np_1000)).reshape(10000,108).float().to(device)  
# convert ndarray into torch tensor
#X_train_tensor = torch.from_numpy(X_train).float().to(device)

# Create class Custom Dataset 
class Fe_Exp_dataset(Dataset):
    def __init__(self):
        # data loading
        self.fe_data = x_fe_tensor # as feature
        self.exp_data = x_exp_tensor # as target
        self.n_sample = x_fe_tensor.shape[0]
        
    def __getitem__(self, index):
        return self.fe_data[index] , self.exp_data[index]
    
    def __len__(self):
        return self.n_sample

# Dataset
train_dataset = Fe_Exp_dataset()
first_data = train_dataset[0]
feature, labels = first_data
print(feature, labels)

# MODEL STRUCTURE
# model 1
class Conv1D_DA1(nn.Module):
    def __init__(self):
        super(Conv1D_DA1, self).__init__()
        self.conv1d_encoder=nn.Sequential(
                nn.Conv1d(in_channels=1, out_channels=16, kernel_size=4,stride=2),
                nn.BatchNorm1d(16),
                nn.Conv1d(16, 32, kernel_size=4, stride= 2),
                nn.BatchNorm1d(32),
                nn.Conv1d(32, 64, kernel_size=4, stride= 2),
                nn.BatchNorm1d(64),
                nn.Conv1d(64, 128, kernel_size=4, stride= 2),
                nn.BatchNorm1d(128)
        )
        self.fc_encoder = nn.Sequential(
                nn.Linear(11*128,1000),
                nn.Tanh()
            )
        self.fc_decoder=nn.Sequential(
                nn.Linear(1000, 11*128),
                nn.Tanh()
        )
        self.conv1d_decoder = nn.Sequential(
                nn.ConvTranspose1d(128, 64, kernel_size=4, stride= 2),
                nn.BatchNorm1d(64),
                nn.ConvTranspose1d(64, out_channels=32, kernel_size=4, stride=2),
                nn.BatchNorm1d(32),
                nn.ConvTranspose1d(32, out_channels=16, kernel_size=4, stride =2),
                nn.BatchNorm1d(16),
                nn.ConvTranspose1d(16, 1, kernel_size=6,stride=2),
                nn.Tanh()
            )
    def forward(self, x):
        x=self.conv1d_encoder(x)
        x = x.view(-1,128*11)
        x = self.fc_encoder(x)
        x = self.fc_decoder(x)
        x = x.view(-1, 128, 11)
        x = self.conv1d_decoder(x)
        return x

    def num_flat_features(self,x):
        size=x.size()[1:] #all dimension except the batch dimension
        num_features=1
        for s in size:
            num_features *=s
        return num_features
# Model 2
class Conv1D_DA2(nn.Module):
    def __init__(self):
        super(Conv1D_DA2, self).__init__()
        self.conv1d_encoder=nn.Sequential(
                nn.Conv1d(in_channels=1, out_channels=16, kernel_size=4,stride=2),
                nn.BatchNorm1d(16),
                nn.Conv1d(16, 32, kernel_size=4, stride= 2),
                nn.BatchNorm1d(32),
                nn.Conv1d(32, 64, kernel_size=4, stride= 2),
                nn.BatchNorm1d(64),
                nn.Conv1d(64, 128, kernel_size=4, stride= 2),
                nn.BatchNorm1d(128)
        )
        self.fc_encoder = nn.Sequential(
                nn.Linear(4*128,108)
            )

    def forward(self, x):
        x=self.conv1d_encoder(x)
        x = x.view(-1,128*4)
        x = self.fc_encoder(x)
        return x

    def num_flat_features(self,x):
        size=x.size()[1:] #all dimension except the batch dimension
        num_features=1
        for s in size:
            num_features *=s
        return num_features
    
# Now we can create a model and send it at once to the device
AE_model = Conv1D_DA2().to(device)
from torchsummary import summary
summary(AE_model,(1,208))
# TRAINING STEP
batch_size = 1000
train_loss = []
def train(AE_model, num_epochs, batch_size, lr):
    torch.manual_seed(42)
    criterion = nn.MSELoss() # mean square error loss
    #optimizer = optim.SGD(AE_model.parameters(), lr=lr,momentum=0.90,weight_decay=0.001)
    optimizer = optim.Adam(AE_model.parameters(),lr, weight_decay=1e-5) 
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    outputs = []
    run_loss = []
    #train_loss = []
    for epoch in range(num_epochs):
        for data in train_loader:
            feature, labels = data
            recon = AE_model(feature)
            loss = criterion(recon, labels)
            run_loss.append(loss.item())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        train_loss.append(np.sum(loss.item()))
        plt.plot(np.log10(train_loss),'r-')
        plt.xlabel('epoch')
        plt.ylabel('log loss')
        if epoch % 10 == 0:
            print('Epoch:{}, Loss:{:.6f}'.format(epoch+1, float(loss)))
        outputs.append((epoch, feature, recon),)
    #return outputs
    plt.show()

lr = 0.002
max_epochs = 51
outputs = train(AE_model, num_epochs=max_epochs, batch_size= batch_size, lr=lr)
# train weight of fc_encoder output layer
# test AE_model
def test_model(x):
    y_hat=AE_model(x)
    y_hat=y_hat.reshape(108,).detach().numpy()
    return y_hat

    
############# Plot original and reconstructed signal   #######################     
for i in range(10):
    x = x_fe_tensor[i].reshape(1,1,108)
    y = test_model(x)
    #print(y)
    plt.plot(y)
plt.show()
    
######################## Model testing with dam data ########################
# Load experimental data
dam_050_265 = np.loadtxt("x_dam_265_050_46Hz_sym2_100.csv", delimiter=",")
dam_100_265 = np.loadtxt("x_dam_265_100_54Hz_sym2_100.csv", delimiter=",")
dam_070_385 = np.loadtxt("x_dam_385_070_46Hz_sym2_100.csv", delimiter=",")
# Select max MSD feature indexes points
dam_050_265_msd = np.zeros([108,1000])
for i,j in enumerate(idx_msd):
    n = dam_050_265[j,:]
    dam_050_265_msd[i,:] = n
dam_050_265 = dam_050_265_msd 

dam_100_265_msd = np.zeros([108,100])
for i,j in enumerate(idx_msd):
    n = dam_100_265[j,:]
    dam_100_265_msd[i,:] = n
dam_100_265 = dam_100_265_msd 

dam_070_385_msd = np.zeros([108,100])
for i,j in enumerate(idx_msd):
    n = dam_070_385[j,:]
    dam_100_265_msd[i,:] = n
dam_070_265 = dam_070_385_msd 
###############################################################################
x_fe_sym_norm = x_fe_sym_norm_msd

# Add noise to FEM damaged data
FE_data_n_3 = np.zeros([108,494])
for i in range(494):
    n_data = add_noise(noise_percent = 4,x=x_fe_sym_norm[0:108,i])
    FE_data_n_3[:,i] = n_data


# plot fe and exp data 
plt.plot(idx_msd, x_fe_sym_norm[:,252],'b.')
plt.plot(idx_msd, dam_100_265[:,8],'r-.')
plt.legend(['FE_simualted','Exp'])
plt.title('100 mm Deb. at 265 mm')
#plt.xlim([180,207])
#plt.ylim([0,0.12])
plt.show()


for i in range(1):
    x_dam = FE_data_n_3[0:108,242] 
    x_dam_tensor = torch.from_numpy(x_dam).reshape(1,1,108).float().to(device)
    x = x_dam_tensor
    y = test_model(x)
    plt.plot(idx_msd, y,'r-.')
plt.plot(idx_msd[1:], dam_050_265[1:,10],'b-+')
plt.legend(['FE_recunstucted','Exp'])
plt.title('100 mm Deb. at 265 mm')
#plt.xlim([180,207])
#plt.ylim([0,0.12])
plt.show()
# Store transformed FE data 
FE_transformed_data = np.zeros([108,494])
for i in range(494):
    x_dam = FE_data_n_3[0:108,i] 
    x_dam_tensor = torch.from_numpy(x_dam).reshape(1,1,108).float().to(device)
    x = x_dam_tensor
    y = test_model(x)
    FE_transformed_data[:,i] = y
    
########################### Dimension reduction PCA ###########################
from sklearn.decomposition import PCA
# Concanate FE and Exp data 
FE_EXP_Data = np.concatenate((FE_transformed_data,dam_050_265[:,0:3]),axis = 1)
FE_EXP_Data_trn = np.transpose(FE_EXP_Data)
FE_transformed_data_tnsp = np.transpose(FE_transformed_data)
pca = PCA(n_components=16, svd_solver='randomized')
X_train_pca = pca.fit_transform(FE_EXP_Data_trn)
# Projection
X_pca_comp = pca.components_

# Plot pca varience ratio
pca_val = pca.explained_variance_ratio_
pca_cum = np.cumsum(pca_val)
plt.bar(range(1,len(pca_val)+1), pca_val, width=0.9, align='center', label='Individual explained variance')
plt.step(range(1,len(pca_cum)+1), pca_cum, where='mid',label='Cumulative explained variance')
for i, v in enumerate(pca_cum):
    plt.text(i+0.5, v + 0.02, '{0:.2f}'.format(v))
plt.ylim((0, 1))
plt.xlim((0,16))
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal component index')
plt.legend(loc='center right')
plt.tight_layout()
plt.show()
###############################################################################
######################### Debonding Prediction Model ##########################
###############################################################################
"""
debonding location from 105 to 425 mm with 20 mm of step size, and size is varies fron 
10 mm to 150 mm with step size of 5 mm 
"""
fe_target_loc = np.zeros([493,1])
for i in range(17):
    for j in range(29):
        t = np.array(105+20*i)
        fe_target_loc[i*29+j] = t
# location target zone wise for clasification
fe_loc_zone_t = np.zeros([493,1])
for i in range(5):
    for j in range(29*3):
        if i == 0:
            t1 =np.array(1)
            fe_loc_zone_t[j] = t1
        elif i == 1:
            t2 =np.array(2)
            fe_loc_zone_t[29*3+j] = t2
        elif i == 2:
            t3 = np.array(3)
            for k in range(29*5):
                fe_loc_zone_t[29*6+k] = t3
        elif i == 3:
            t4 =np.array(4)
            fe_loc_zone_t[29*11+j] = t4
        elif i == 4:
            t5 = np.array(5)
            fe_loc_zone_t[29*14+j] = t5
            
fe_loc_zone_t = fe_loc_zone_t.reshape(493)

# size target zone wise for clasification
fe_size_zone_t = np.zeros([493,1])
for i in range(17):
    for j in range(7):
        if j == 0:
            t =np.array(1)
            for k in range(5):
                fe_size_zone_t[29*i+k] = t           
        else:
            for k in range(4):
                t2 =np.array(j+1)
                fe_size_zone_t[29*i+5+k+(j-1)*4] = t2

fe_size_zone_t = fe_size_zone_t.reshape(493)

fe_target_loc  = fe_target_loc.reshape(493)  
fe_target_size = np.zeros([1,493])
for i in range(17):
    for j in range(29):
        t = np.array(10+j*5)
        fe_target_size[:,i*29+j] =t
fe_target_size = fe_target_size.reshape(493)
####################### RandomForestClass #####################################
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
# x train and test data 
#X_train = np.concatenate((X_train_pca[0:232,:],X_train_pca[233:494,:]))
X_train = np.concatenate((FE_EXP_Data_trn[0:232,:],FE_EXP_Data_trn[233:494,:]))
x_trn, x_tst, y_trn, y_tst = train_test_split(X_train,fe_loc_zone_t, 
                                              test_size= 0.15, random_state=(0))
# Create a base model
reg_rf = RandomForestRegressor()
cls_rf = RandomForestClassifier()
print(reg_rf.get_params())
# Hyperparameter tuning using Gridsearch
# Create parameter grid
rf_param_grid = {
    'bootstrap' : [True],
    'max_depth' : [80, 90, 100, 110],
    'n_estimators' : [100, 150, 200, 250, 300]
    }
# Intant gread search model
rf_grid_search = GridSearchCV(estimator=cls_rf, param_grid=rf_param_grid)
# Fit grid search to data
rf_grid_search.fit(x_trn, y_trn)
print(rf_grid_search.best_params_)
print(rf_grid_search.score(x_tst, y_tst))
reg_rf.fit(x_trn,y_trn)

rf_grid_search.predict(FE_EXP_Data_trn[494:497])
rf_pred = rf_grid_search.predict(x_tst)
print(rf_grid_search.score(x_trn, y_trn))
## Plot result
plt.plot(rf_pred,'b.')
plt.plot(y_tst,'r.')
plt.show()
################################### SVM #######################################
from sklearn.svm import SVC
rng = np.random.RandomState(0)


reg_svm = SVC()
#Create a svm Classifier and hyper parameter tuning 
# defining parameter range
param_grid = {'C': [ 1, 10, 100, 1000,10000], 
              'gamma': [1,0.1,0.01,0.001,0.0001],
              'kernel': ['rbf']} 
  
grid_svm = GridSearchCV(reg_svm, param_grid)
# fitting the model for grid search
grid_search_svm=grid_svm.fit(x_trn, y_trn)
print(grid_search_svm.best_params_)

#clf.fit(X_trn, y_trn)
y_svm_pred = grid_svm.predict(x_tst)
y = y_tst
print(r2_score(y_trn, grid_svm.predict(x_trn)))
# Plot result
plt.plot(y_svm_pred,'b.')
plt.plot(y, 'r.')
plt.show()
"""
#save and the train_model
#file_model = 'Conv1D_AE.pth' 
#torch.save(Conv1D_AE, file_model) # Save model structure
# file = 'AE_CF_4_snr55_60_65-2.pth'  #trained model file 
# torch.save(AE_model.state_dict(), file)  #saving trained model
# Load model 
#AE_model_trained = Conv1D_AE()
#AE_model_trained.load_state_dict(torch.load(file))
#extacting weight as feature for classifier
activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook
AE_model.fc_encoder.register_forward_hook(get_activation('fc2'))
x = torch.randn(1, 1, 41)
output = AE_model(x)
print(activation['fc2'])
for j in range(1000):
    for i in x_train_tensor[j]:
        AE_model.fc_encoder.register_forward_hook(get_activation('fc_encoder'))
        x = i.view(1,1,41)
        output = AE_model(x)
        #print(activation['fc2'])
        plt.plot(activation['fc_encoder'][0].numpy())
plt.show()
"""
###############################  MAHALANOBIS  #################################
