import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import scipy.stats as stats

############################### Load Exp data #################################
#%%
x_fe = pd.read_excel("D:/Vibration_Model_stiffplate/Domain_Adapt/FE_data_105_10_to_425_150.xlsx",header = None) 
x_fe_np = x_fe.to_numpy() 
x_fe_sym_np = x_fe_np[0:208,:]
# normalize(0-1) data
x_fe_sym_norm = (x_fe_sym_np - np.min(np.min(x_fe_sym_np)))/(np.max(np.max(x_fe_sym_np))-np.min(np.min(x_fe_sym_np))) 
x_fe_sym_norm_dam = np.concatenate((x_fe_sym_norm[:,0:232],x_fe_sym_norm[:,233:494]),axis=1)
# Save as Matlab file
"""
from scipy.io import savemat
a = np.arange(20)
mdic = {"data": x_fe_sym_norm, "label": "Fe_data_sym"}
savemat("Fe_data_sym.mat", mdic)
"""
#%%

###############################################################################
x_undam_fe_sym = x_fe_sym_norm[0:208,232]
x_undam =[]# np.empty([416,1])
for i in range(35):
    file = 'D:/Vibration_Model_stiffplate/Domain_Adapt/New_Exp_2022/plate_dim_330x530_dam_at_265_000/Scan_'+str(i+1)+'_46_Hz_disp_abs.xlsx'
    x = pd.read_excel(file).to_numpy()
    x_d = x[10:426,4]
    x_undam.append(x_d)
x_undam_np = np.array(x_undam)
x_undam_np_sym1 = x_undam_np[:,0:208]
x_undam_np_sym2 = x_undam_np[:,208:416]
# Flip the sym2 data to make like sym1 data
x_undam_np_sym2 = np.flip(x_undam_np_sym2) 
# Plot Data
for i in range(33):
    plt.plot(x_undam_np_sym2[i+2,:])
plt.show()
######################## Normalize Exp data ###################################
x_undam_np_sym1_norm = np.zeros([35,208])
for i in range(35):
    for j in range(208):
        n = (x_undam_np_sym1[i,j]-np.min(x_undam_np_sym1[i,:]))/(np.max(x_undam_np_sym1[i,:])-np.min(x_undam_np_sym1[i,:]))
        x_undam_np_sym1_norm[i,j] = n

x_undam_np_sym2_norm = np.zeros([35,208])        
for i in range(35):
    for j in range(208):
        n = (x_undam_np_sym2[i,j]-np.min(x_undam_np_sym2[i,:]))/(np.max(x_undam_np_sym2[i,:])-np.min(x_undam_np_sym2[i,:]))
        x_undam_np_sym2_norm[i,j] = n
# Concanate Sym1 and Sym2 norm data to make larger dataset
x_undam_np_sym_norm = np.concatenate((x_undam_np_sym1_norm,x_undam_np_sym2_norm), axis=0)
# Plot Normalizzed Data
for i in range(67):
    plt.plot(x_undam_np_sym_norm[i+2,:])
plt.show()       
# Pandas DataFrame
column = list(np.arange(1,209))
undam_x_df = pd.DataFrame(x_undam_np_sym2_norm,columns=column)

# Corelation Matrix
co_rel = undam_x_df.corr()
plt.imshow(co_rel)
plt.show()

########### bootstrapping of all column data in loop ##########################
undam_bst_data = []   # Empty List
for j in range(208):
    bst = pd.DataFrame({str(j+1):[undam_x_df.sample(20,replace=True)[j+1].mean() for i in range(0,200)]})
    undam_bst_data.append(bst)

# List to data frame in loop by adding column
undam_bst_df = pd.DataFrame() #Empty DataFrame
for i in range(208): 
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
for i in range(208):
    m = undam_x_df[i+1].mean()
    undam_x_mean_df.append(m)
    
undam_x_std_df = []
for i in range(208):
    sd = undam_x_df[i+1].std()
    undam_x_std_df.append(sd)

##################### estimete mean and sd of bst_df ##########################
undam_mean_df = []
for i in range(208):
    m = undam_bst_df[i+1].mean()
    undam_mean_df.append(m)
    
undam_std_df = []
for i in range(208):
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
#%%
import random
undam_norm_df_1000 = []
for i in range(208):
    norm_d = [random.gauss(undam_mean_df[i],undam_std_df[i]) for j in range(1000)]
    undam_norm_df_1000.append(norm_d)
undam_norm_df_np_1000 = np.array(undam_norm_df_1000)
#%%
################# Plot Generated data #########################################
for i in range(50):
    plt.plot(undam_norm_df_np_1000[:,i])
plt.plot(x_undam_fe_sym,'r-')
plt.show()
##################### ADD Noise to undam FE data ##############################
# Function to add % noise
def add_noise(noise_percent, x):
    r = np.random.normal(0,1,size=(x.shape))
    x_noise = x*(1+(noise_percent/100)*r)
    return x_noise
    
# generate the noisy data
#%%
undam_fe_n_data = np.zeros([208,1000])
for i in range(1000):
    n_data = add_noise(noise_percent = 0.1,x=x_undam_fe_sym)
    undam_fe_n_data[:,i] = n_data
#%%    
# Plot noisy data 
for i in range(100):
    plt.plot(undam_fe_n_data[:,i])
    #plt.plot(undam_norm_df_np_1000[:,i])
plt.show()
#%%
# Load experimental data
dam_050_265 = np.loadtxt("x_dam_265_050_46Hz_sym2_100.csv", delimiter=",")
dam_050_265 = dam_050_265[:,0:100]
dam_100_265 = np.loadtxt("x_dam_265_100_54Hz_sym2_100.csv", delimiter=",")
dam_070_385 = np.loadtxt("x_dam_385_070_46Hz_sym2_100.csv", delimiter=",")
#%%
################### Transform Fe data and exp data PCA #########################
# Cancatnate FE data and Exp data in single matrix
FE_EXP_DATA_ALL = np.concatenate((undam_fe_n_data,undam_norm_df_np_1000,x_fe_sym_norm_dam,dam_050_265,dam_100_265,dam_070_385),axis=1)
###################################################################################
############################### Dimension reduction PCA ###########################
###################################################################################
#%%
from sklearn.decomposition import PCA
# Concanate FE and Exp data 
#FE_EXP_Data = np.concatenate((FE_transformed_data,dam_050_265[:,0:3]),axis = 1)
FE_EXP_Data = FE_EXP_DATA_ALL
FE_EXP_Data_trn = np.transpose(FE_EXP_Data)
pca = PCA(n_components=5, svd_solver='randomized')
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
#%%
# Normalize PCA data
X_train_pca_norm = (X_train_pca - np.min(np.min(X_train_pca)))/(np.max(np.max(X_train_pca))-np.min(np.min(X_train_pca))) 
#%%
plt.plot(X_train_pca_norm[2,:])
plt.plot(X_train_pca_norm[2502,:])
plt.plot(X_train_pca_norm[2702,:])
plt.plot(X_train_pca_norm[2703,:])
plt.plot(X_train_pca_norm[2704,:])
plt.plot(X_train_pca_norm[2602,:])
plt.show()
#%%
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
#%%
x_fe_tensor = torch.from_numpy(np.transpose(X_train_pca_norm[0:1000])).reshape(1000,5).float().to(device)
x_exp_tensor = torch.from_numpy(np.transpose(X_train_pca_norm[1000:2000])).reshape(1000,5).float().to(device)  
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
#print(feature, labels)
from torch.utils.data import random_split
data_len = len(train_dataset)
split = int(data_len*.8)
X_train_undam,X_test_undam = random_split(train_dataset,[split,data_len-split],generator=torch.Generator().manual_seed(42))
#%%
# MODEL STRUCTURE
# model 1
class Conv1D_DA1(nn.Module):
    def __init__(self):
        super(Conv1D_DA1, self).__init__()
        self.fc_1=nn.Sequential(
            nn.Linear(in_features=5, out_features=6),
            nn.Tanh(),
            nn.Linear(6, 5)
            )

    def forward(self, x):
        x=self.fc_1(x)
        return x

    def num_flat_features(self,x):
        size=x.size()[1:] #all dimension except the batch dimension
        num_features=1
        for s in size:
            num_features *=s
        return num_features
# Now we can create a model and send it at once to the device
AE_model = Conv1D_DA1().to(device)
from torchsummary import summary
summary(AE_model,(1,5))
# TRAINING STEP
#%%
batch_size = 1000

def train(AE_model, num_epochs, batch_size, lr):
    train_loss = []
    test_loss = []
    torch.manual_seed(42)
    criterion = nn.MSELoss() # mean square error loss
    #optimizer = optim.SGD(AE_model.parameters(), lr=lr,momentum=0.90,weight_decay=0.001)
    optimizer = optim.Adam(AE_model.parameters(),lr=lr, weight_decay=1e-5) 
    #train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    train_loader = DataLoader(X_train_undam, batch_size = batch_size, shuffle=True)
    test_loader = DataLoader(X_test_undam, batch_size=batch_size, shuffle=True)
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
        # Run the testing batches
        with torch.no_grad():
            for b, (X_test, y_test) in enumerate(test_loader):
                # Apply the model
                y_val = AE_model(X_test)
        loss_t = criterion(y_val, y_test)
        test_loss.append(np.sum(loss_t.item())) 
        if epoch % 1 == 0:
            print('Epoch:{}, train loss:{:.8f}'.format(epoch+1, float(loss)))
        outputs.append((epoch, feature, recon),)
        plt.plot(np.log10(train_loss), 'r-')
        plt.plot(np.log10(test_loss),'b-')
        plt.xlabel('epoch')
        plt.ylabel('log loss')
        plt.legend(['training loss','validation loss'])
    #return outputs
    plt.show()

lr = 0.02
max_epochs = 101
outputs = train(AE_model, num_epochs=max_epochs, batch_size= batch_size, lr=lr)
#%%
# test AE_model
def test_model(x):
    y_hat=AE_model(x)
    y_hat=y_hat.reshape(5,).detach().numpy()
    return y_hat

#%%
############# Plot original and reconstructed signal   #######################     
for i in range(10):
    x = x_fe_tensor[i].reshape(1,5)
    y = test_model(x)
    #print(y)
    plt.plot(y)
plt.show()
#%%
###############################################################################
# Add noise to FEM damaged data
X_fe_train_pca_norm = np.transpose(X_train_pca_norm[2000:2493,:])
FE_data_n_3 = np.zeros([5,493])
for i in range(493):
    n_data = add_noise(noise_percent = 0.1,x=X_fe_train_pca_norm[:,i])
    FE_data_n_3[:,i] = n_data
#%%
for i in range(400):
    x_dam = FE_data_n_3[0:208,i] 
    x_dam_tensor = torch.from_numpy(x_dam).reshape(1,5).float().to(device)
    x = x_dam_tensor
    y = test_model(x)
    plt.plot(y)
plt.plot(X_train_pca_norm[4,:],'r-.')
plt.legend(['FE_recunstucted','Exp'])
plt.title('70 mm Deb. at 385 mm')
#plt.xlim([180,207])
#plt.ylim([0,0.12])
plt.show()
#%%
# Store transformed FE data 
FE_transformed_data = np.zeros([5,493])
for i in range(493):
    x_dam = FE_data_n_3[:,i] 
    x_dam_tensor = torch.from_numpy(x_dam).reshape(1,5).float().to(device)
    x = x_dam_tensor
    y = test_model(x)
    FE_transformed_data[:,i] = y
#%%

