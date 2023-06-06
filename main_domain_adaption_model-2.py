import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import scipy.stats as stats

############################### Load data #################################
#%%
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
idx2 = np.array(range(17,25))
idx3 = np.array(range(30,40))
idx4 = np.array(range(42,51))
idx5 = np.array(range(55,65))
idx6 = np.array(range(67,77))
idx7 = np.array(range(91,102))
idx8 = np.array(range(116,129))
idx9 = np.array(range(142,154))
idx10 = np.array(range(169,182))

idx_msd = []
for i in range(10):
    file = 'idx'+str(i+1)
    file = eval(file)  # eval is a method to convert str to object
    idx_msd.extend(file)
#%%
###############################################################################
x_fe_sym_norm_msd = np.zeros([108,494])
for i,j in enumerate(idx_msd):
    n = x_fe_sym_norm[j,:]
    x_fe_sym_norm_msd [i,:] = n

x_fe_sym_norm = x_fe_sym_norm

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
# pull out the value coresponding to max MSD
x_undam_np_sym1_msd = np.zeros([35,len(idx_msd)])
for i,j in enumerate(idx_msd):
    n = x_undam_np_sym1[:,j]
    x_undam_np_sym1_msd[:,i] = n
x_undam_np_sym1 = x_undam_np_sym1
x_undam_np_sym2 = x_undam_np[:,208:416]
# Flip the sym2 data to make like sym1 data
x_undam_np_sym2 = np.flip(x_undam_np_sym2)
# pull out the value coresponding to max MSD
x_undam_np_sym2_msd = np.zeros([35,len(idx_msd)])
for i,j in enumerate(idx_msd):
    n = x_undam_np_sym2[:,j]
    x_undam_np_sym2_msd[:,i] = n

x_undam_np_sym2 = x_undam_np_sym2  
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
import random
undam_norm_df_1000 = []
for i in range(208):
    norm_d = [random.gauss(undam_mean_df[i],undam_std_df[i]) for j in range(1000)]
    undam_norm_df_1000.append(norm_d)

################# Plot Generated data #########################################
# List to np array
undam_norm_df_np_1000 = np.array(undam_norm_df_1000)
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
x_fe_tensor = torch.from_numpy(np.transpose(undam_fe_n_data)).reshape(1000,208).float().to(device)
x_exp_tensor = torch.from_numpy(np.transpose(undam_norm_df_np_1000)).reshape(1000,208).float().to(device)  
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
split = int(data_len*.9)
X_train_undam,X_test_undam = random_split(train_dataset,[split,data_len-split],generator=torch.Generator().manual_seed(42))
#%%
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
                nn.ConvTranspose1d(16, 1, kernel_size=6,stride=2)
            )
    def forward(self, x):
        x = x.reshape(-1,1,208)
        x=self.conv1d_encoder(x)
        x = x.view(-1,128*11)
        x = self.fc_encoder(x)
        x = self.fc_decoder(x)
        x = x.view(-1, 128, 11)
        x = self.conv1d_decoder(x)
        return x.reshape(-1, 208)

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
                nn.Linear(11*128,1024),
                nn.Tanh(),
                nn.Linear(1024, 512),
                nn.Tanh(),
                nn.Linear(512, 208)
            )

    def forward(self, x):
        x = x.reshape(-1,1,208)
        x=self.conv1d_encoder(x)
        x = x.view(-1,128*11)
        x = self.fc_encoder(x)
        return x

    def num_flat_features(self,x):
        size=x.size()[1:] #all dimension except the batch dimension
        num_features=1
        for s in size:
            num_features *=s
        return num_features
# Model 3
class DNN_DA3(nn.Module):
    def __init__(self):
        super(DNN_DA3, self).__init__()
        self.fc_layer = nn.Sequential(
            nn.Linear(208, 1024),
            nn.Tanh(),
            nn.Linear(1024, 512),
            nn.Tanh(),
            nn.Linear(512, 208)
        )
    def forward(self, x):
        x = self.fc_layer(x)
        return x
    def num_flat_features(self,x):
        size=x.size()[1:] #all dimension except the batch dimension
        num_features=1
        for s in size:
            num_features *=s
        return num_features
# Now we can create a model and send it at once to the device
#%%
AE_model = DNN_DA3().to(device)
#AE_model = Conv1D_DA2().to(device)
from torchsummary import summary
summary(AE_model,(1,208))
# TRAINING STEP
#%%
batch_size = 100
def train(model, num_epochs, batch_size, lr):
    print(str(model.__class__.__name__))
    train_loss = []
    test_loss = []
    early_stop_counter = 0
    early_stop_threshold = 10
    best_val_loss = float('inf')
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
        if loss_t < best_val_loss:
        #if loss_t > 0.000022:
            best_val_loss = loss_t
            early_stop_counter = 0
        else :
            early_stop_counter += 1  
        if early_stop_counter == early_stop_threshold:
            print('Early Stop')
            break
        test_loss.append(np.sum(loss_t.item())) 
        if epoch % 10 == 0:
            print('Epoch:{}, train loss:{:.8f}'.format(epoch+1, float(loss)))
        outputs.append((epoch, feature, recon),)
        plt.plot(np.log10(train_loss), 'r-')
        plt.plot(np.log10(test_loss),'b-')
        plt.xlabel('epoch')
        plt.ylabel('log loss')
        plt.title(str(model.__class__.__name__))
        plt.legend(['training loss','validation loss'])
    #return outputs
    plt.show()
#%% training the model saving trained weight
models = [DNN_DA3().to(device), Conv1D_DA2().to(device), Conv1D_DA1().to(device)]
for model in models:
    lr = 0.0003
    max_epochs = 2001
    batch_size = 1000
    AE_model = model
    train(model, num_epochs=max_epochs, batch_size= batch_size, lr=lr)
    file_model = str(model.__class__.__name__)+'3.pth'
    torch.save(model,file_model) # Save model structure
    file = 'train_'+str(model.__class__.__name__)+'2.pth' #trained model file 
    torch.save(model.state_dict(),file)
#%%
# test AE_model
def test_model(x):
    y_hat=AE_model(x)
    y_hat=y_hat.reshape(208,).detach().numpy()
    return y_hat
######################## Model testing with dam data ########################
#%%
# Load debonding experimental data
dam_050_265 = np.loadtxt("x_dam_265_050_46Hz_sym2_100.csv", delimiter=",")
dam_100_265 = np.loadtxt("x_dam_265_100_54Hz_sym2_100.csv", delimiter=",")
dam_070_385 = np.loadtxt("x_dam_385_070_46Hz_sym2_100.csv", delimiter=",")
#%% # reconstruction error for experimental data representation
models = [Conv1D_DA1().to(device), Conv1D_DA2().to(device), DNN_DA3().to(device)]
for j, model in enumerate(models):
    model = model
    file = 'train_'+str(model.__class__.__name__)+'2.pth'
    model.load_state_dict(torch.load(file))
    AE_model = model
    criterion = nn.MSELoss()
    recon_losses1 = []
    recon_losses2 = []
    recon_losses3 = []
    for i in range(3):
        if i == 0:
            x = x_fe_sym_norm[0:208,242]
            y = dam_050_265[0:208, 10]            
        elif i == 1:
            x = x_fe_sym_norm[0:208, 252]
            y = dam_100_265[0:208, 10]
        elif i == 2:
            x = x_fe_sym_norm[0:208, 242]
            y = dam_050_265[0:208, 10]
        with torch.no_grad():
            x_tensor = torch.from_numpy(x).reshape(1,1,208).float().to(device)
            y_tensor = torch.from_numpy(y).float().to(device)
            y_hat = AE_model(x_tensor)
            y_hat = y_hat.reshape(208,)
            recon_loss = criterion(y_hat,y_tensor)
            print(recon_loss)
        if j == 0:
            recon_losses1.append(recon_loss.item())
        elif j == 1:
            recon_losses2.append(recon_loss.item())
        elif j == 2:
            recon_losses3.append(recon_loss.item())
    if j == 0:
        plt.plot(j+1,np.sum(recon_losses1)/3,'rd')
    elif j == 1:
        plt.plot(j+1,np.sum(recon_losses2)/3,'bd')
    elif j == 2:
        plt.plot(j+1,np.sum(recon_losses3)/3,'gd')
    plt.xticks([1,2,3])
    plt.xlabel('model')
    plt.ylim([0.0028,0.0029])
    plt.ylabel('Avg. Construction Error')
plt.show()
#%% Visual comparison of model in reconstruction the exp signal
for i, model in enumerate(models):
    model = model
    file = 'train_'+str(model.__class__.__name__)+'2.pth'
    model.load_state_dict(torch.load(file))
    AE_model = model
    x = x_fe_sym_norm[0:208,242]
    y = dam_050_265[0:208, 10] 
    x_tensor = torch.from_numpy(x).reshape(1,1,208).float().to(device)
    y_hat = AE_model(x_tensor)
    y_hat= y_hat.reshape(208,).detach().numpy()
    #plt.plot(y,'b-')
    if i == 0:
        plt.plot(y_hat,'r-.',label ='model-1')
    elif i == 1:
        plt.plot(y_hat,'g-.',label ='model-2')
    elif i == 2:
        plt.plot(y_hat,'c-.', label = 'model-3')
    #plt.legend('model-1','model-2','model-3')
plt.show()
#%%
###############################################################################
# Add noise to FEM damaged data
FE_data_n_3 = np.zeros([208,494])
for i in range(494):
    n_data = add_noise(noise_percent = 1,x=x_fe_sym_norm[0:208,i])
    FE_data_n_3[:,i] = n_data
#%%
# Store transformed FE data 
models = [Conv1D_DA1().to(device), Conv1D_DA2().to(device), DNN_DA3().to(device)]
FE_transformed_data = np.zeros([208,494])
for i in range(494):
    x_dam = FE_data_n_3[0:208,i] 
    x_dam_tensor = torch.from_numpy(x_dam).reshape(1,1,208).float().to(device)
    x = x_dam_tensor
    model = Conv1D_DA1().to(device)
    file = 'train_'+str(model.__class__.__name__)+'2.pth'
    model.load_state_dict(torch.load(file))
    AE_model = model
    y = AE_model(x)
    FE_transformed_data[:,i] = y.detach().numpy()
#%%
###################################### AE_model Comparision ###################################
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

# damage location and size target value
# location target zone wise for clasification
#%%
fe_loc_zone_t = np.zeros([493,1])
for i in range(5):
    for j in range(29*3):
        if i == 0:
            t1 =np.array(0)
            fe_loc_zone_t[j] = t1
        elif i == 1:
            t2 =np.array(1)
            fe_loc_zone_t[29*3+j] = t2
        elif i == 2:
            t3 = np.array(2)
            for k in range(29*5):
                fe_loc_zone_t[29*6+k] = t3
        elif i == 3:
            t4 =np.array(3)
            fe_loc_zone_t[29*11+j] = t4
        elif i == 4:
            t5 = np.array(4)
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
#%% 
################################################################################################
#Transformation Model sensitivity Analysis by measuring the debonding assesment accuracy of ML # 
################################################################################################
# Testin loop
models = [DNN_DA3().to(device), Conv1D_DA2().to(device), Conv1D_DA1().to(device)]
for j, model in enumerate(models):
    #Load model 
    model = model
    file = 'train_'+str(model.__class__.__name__)+'2.pth'
    model.load_state_dict(torch.load(file))
    # train model
    fe_data_n = np.zeros([208, 494])
    for i in range(494):
        n_data = add_noise(noise_percent = 0.1,x=x_fe_sym_norm[0:208,i])
        fe_data_n[:,i] = n_data
    fe_trans_data = np.zeros([208, 494])
    AE_model = model
    for i in range(494):
        dam = fe_data_n[0:208,i]
        dam_tensor = torch.from_numpy(dam).reshape(1,1,208).float().to(device)
        y = test_model(dam_tensor)
        fe_trans_data[:,i] = y
    FE_EXP_Data = np.concatenate((fe_trans_data,dam_050_265[:,0:10],dam_100_265[:,0:10],dam_070_385[:,0:10]),axis = 1)
    FE_EXP_Data_trns = np.transpose(FE_EXP_Data)
    pca = PCA(n_components = 7, svd_solver = 'randomized')
    x_trn_pca = pca.fit_transform(FE_EXP_Data_trns)
    x_trn_pca_nrm = (x_trn_pca - np.min(np.min(x_trn_pca)))/(np.max(np.max(x_trn_pca))-np.min(np.min(x_trn_pca)))
    fe_trnf_dam = np.concatenate((x_trn_pca_nrm[0:232,:],x_trn_pca_nrm[232:493,:]),axis=0)  
    x_test = fe_loc_zone_t
    trn_tst_split = [0.25, 0.2, 0.15, 0.1]
    train_score = []
    test_score = []
    for split in trn_tst_split:
        x_trn, x_tst, y_trn, y_tst = train_test_split(fe_trnf_dam,x_test, test_size= split, random_state=(0))
        param_svm = {'C': [ 1, 10, 100, 1000,10000], 
                      'gamma': [1,0.1,0.01,0.001,0.0001],
                      'kernel': ['rbf']} 
        svm = SVC()
        grid_svm = GridSearchCV(svm, param_svm)
        grid_svm.fit(x_trn, y_trn)
        score_tr = grid_svm.score(x_trn, y_trn)
        score_ts = grid_svm.score(x_tst, y_tst)
        train_score.append(score_tr)
        test_score.append(score_ts)
    # Set position of bar on x-axis
    x_ax = np.array(trn_tst_split)*100
    if j == 0:
        plt.plot(x_ax, test_score,'r-')
    elif j == 1:
        plt.plot(x_ax, test_score,'b-')
    elif j == 2:
        plt.plot(x_ax, test_score,'g-')
    plt.xlabel('% test dataset')
    plt.ylabel('Score')
    #plt.ylim([0.7,1])
    plt.legend(['model-1','model-2', 'model-3'],loc = 'best')
    plt.xticks(x_ax)
plt.show()     
#%%
#print(f'Model: {str(model.__class__.__name__)}, test: {split*100}%, SVM train Score: {grid_svm.score(x_trn,y_trn):.4f}')
#print(f'Model: {str(model.__class__.__name__)}, test: {split*100}% SVM test Score: {grid_svm.score(x_tst,y_tst):.4f}')
###################################################################################
############################### Dimension reduction PCA ###########################
###################################################################################
# Select max MSD feature indexes points
dam_050_265 = dam_050_265[:,0:100]
dam_050_265_msd = np.zeros([108,100])
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
    dam_070_385_msd[i,:] = n
dam_070_385 = dam_070_385_msd 
#%%
#Load model 
model = Conv1D_DA1().to(device)
file = 'train_'+str(model.__class__.__name__)+'2.pth'
model.load_state_dict(torch.load(file))
fe_data_n = np.zeros([208, 494])
for i in range(494):
    n_data = add_noise(noise_percent = 1,x=x_fe_sym_norm[0:208,i])
    fe_data_n[:,i] = n_data
fe_trans_data = np.zeros([208, 494])
AE_model = model
for i in range(494):
    dam = fe_data_n[0:208,i]
    dam_tensor = torch.from_numpy(dam).reshape(1,1,208).float().to(device)
    y = test_model(dam_tensor)
    fe_trans_data[:,i] = y
#%%
# Concanate FE and Exp data 
FE_EXP_Data = np.concatenate((undam_norm_df_np_1000[:,0:100]
                              ,fe_trans_data,dam_050_265[:,0:100],dam_100_265[:,0:100],dam_070_385[:,0:100]),axis = 1)
FE_EXP_Data = np.concatenate((fe_trans_data,dam_050_265[:,0:100],dam_100_265[:,0:100],dam_070_385[:,0:100]),axis = 1)
#FE_EXP_Data = fe_trans_data
#FE_EXP_Data = fe_trans_data
FE_EXP_Data_trn = np.transpose(FE_EXP_Data)
pca = PCA(n_components=16, svd_solver='randomized')
X_train_pca = pca.fit_transform(FE_EXP_Data_trn)
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
FE_tranf_undam = X_train_pca_norm[232,:]

FE_tranf_dam = np.concatenate((X_train_pca_norm[0:232,:],X_train_pca_norm[232:493,:]),axis=0)    
#EXP_000_000 = X_train_pca_norm[0:100,:]
EXP_050_265 = X_train_pca_norm[493:593,:]
EXP_100_265 = X_train_pca_norm[593:693,:]
EXP_070_385 = X_train_pca_norm[693:793,:]
#%%
#%% plot pca componenet
fig = plt.figure()
ax = plt.axes(projection ="3d")
n1 = 0
n2 = 1
n3 = 3
ax.scatter3D(X_train_pca_norm[10:100,n1],X_train_pca_norm[10:100,n2],X_train_pca_norm[10:100,n3], color = 'red')
ax.scatter3D(X_train_pca_norm[100:150,n1],X_train_pca_norm[100:150,n2],X_train_pca_norm[100:150,n3],color='green')
ax.scatter3D(X_train_pca_norm[150:400,n1],X_train_pca_norm[150:400,n2],X_train_pca_norm[150:400,n3],color='blue')
#ax.scatter3D(X_train_pca_norm[494:594,n1],X_train_pca_norm[494:594,n2],X_train_pca_norm[494:594,n3],color='yellow')
#ax.scatter3D(X_train_pca_norm[694:794,n1],X_train_pca_norm[694:794,n2],X_train_pca_norm[694:794,n3],color='black')
plt.xlabel('First Component')
plt.ylabel('Second Component')
ax.set_zlabel('Third Component')
plt.legend(['FE simulation', 'Exp.'])
plt.show()
#%% ######################### PCA component comparirion of exp and FEA ##########################
x_fe_265_50_pca = X_train_pca_norm[242,:]
x_fe_265_100_pca = X_train_pca_norm[252,:]
x_fe_385_70_pca = X_train_pca_norm[420,:]
# add noise to data
X_exp_pca = [x_fe_265_50_pca,x_fe_265_100_pca,x_fe_385_70_pca]
exp_pca = np.zeros([150,16])
for i,x_pca in enumerate(X_exp_pca):
    for j in range(50):
        n_data = add_noise(noise_percent = 10,x=x_pca)
        exp_pca[i*50+j,:] = n_data

fig = plt.figure(figsize=(7,7))
ax = plt.axes(projection ="3d")
n1 = 0
n2 = 1
n3 = 2
ax.scatter3D(exp_pca[0:50,n1],exp_pca[0:50,n2],exp_pca[0:50,n3], color = 'red')
ax.scatter3D(exp_pca[50:100,n1],exp_pca[50:100,n2],exp_pca[50:100,n3], color = 'blue')
ax.scatter3D(exp_pca[100:150,n1],exp_pca[100:150,n2],exp_pca[100:150,n3], color = 'green')
ax.scatter3D(x_fe_265_50_pca[n1],x_fe_265_50_pca[n2],x_fe_265_50_pca[2],color = 'cyan',marker = "s",s=100)
ax.scatter3D(x_fe_265_100_pca[n1],x_fe_265_100_pca[n2],x_fe_265_100_pca[2],c = 'yellow',marker = "d",s=100)
ax.scatter3D(x_fe_385_70_pca[n1],x_fe_385_70_pca[n2],x_fe_385_70_pca[2],c ='magenta',marker = "v",s=100)
plt.xlabel('First Component')
plt.ylabel('Second Component')
ax.set_zlabel('Third Component')
plt.legend(['exp 50 mm at 265', 'exp 100 mm at 265', 'exp 70 mm at 385','FE 50 mm at 265', 'FE 100 mm at 265', 'FE 70 mm at 385'])
plt.show()
#%% pca inverse transform
x_exp_proj = pca.inverse_transform(exp_pca)
plt.plot(x_exp_proj[0,:])
plt.show()
#%%
####################### RandomForestClass #####################################
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
# x train and test data 
#X_train = np.concatenate((X_train_pca[0:232,:],X_train_pca[233:494,:]))
#%%
X_train = FE_tranf_dam
X_test = fe_loc_zone_t
x_trn, x_tst, y_trn, y_tst = train_test_split(
    X_train,X_test, 
    test_size= 0.15, random_state=(0))
#%%
cls_rf = RandomForestClassifier(random_state=42)
# Hyperparameter tuning using Gridsearch
# Create parameter grid
rf_param_grid = {
    'bootstrap' : [True],
    'max_depth' : [80, 90, 100, 110],
    'n_estimators' : [100, 150, 200, 250, 300]
    }
# Intant gread search model
grid_rf = GridSearchCV(estimator=cls_rf, param_grid=rf_param_grid, cv =6)
# Fit grid search to data
grid_rf.fit(x_trn, y_trn)
print(grid_rf.best_params_)
print(grid_rf.score(x_trn, y_trn))
print(grid_rf.score(x_tst, y_tst))
#rf_grid_search.predict(FE_EXP_Data_trn[494:497])
y_rf_pred = grid_rf.predict(x_tst)
y_rf_prob = grid_rf.predict_proba(x_tst)
## Plot result
con_mat=confusion_matrix(y_tst,y_rf_pred)
#print(con_mat)
#class_name = ['group 1','group 2','group 3','group 4','group 5','group 6','group 7']
class_name = ['zone 1','zone 2','zone 3','zone 4','zone 5']
fig, ax=plot_confusion_matrix(conf_mat=con_mat,
                              colorbar=True,
                              show_absolute=True,
                              show_normed=True,
                              class_names=class_name,
                              figsize=[8, 8])
plt.xlabel('predicted lavel', fontweight = 'bold',fontsize = 10)
plt.ylabel('true lavel', fontweight = 'bold', fontsize = 10)
plt.title('rfc',fontweight = 'bold',fontsize = 10)
plt.show()
#%%
################################### SVM #######################################
from sklearn.svm import SVC
rng = np.random.RandomState(0)
svm = SVC(random_state=42,probability=True)
#Create a svm Classifier and hyper parameter tuning 
# defining parameter range
param_svm = {'C': [ 1, 10, 100, 1000,10000], 
              'gamma': [1,0.1,0.01,0.001,0.0001],
              'kernel': ['rbf']} 
  
grid_svm = GridSearchCV(svm, param_svm, cv=6)
# fitting the model for grid search
grid_search_svm=grid_svm.fit(x_trn, y_trn)
print(grid_search_svm.best_params_)
print('SVM train Score',grid_svm.score(x_trn,y_trn))
print('SVM test Score',grid_svm.score(x_tst,y_tst))
y_svm_pred = grid_svm.predict(x_tst)
y_svm_prob = grid_svm.predict_proba(x_tst)
# Plot result
con_mat=confusion_matrix(y_tst,y_svm_pred)
#print(con_mat)
fig, ax=plot_confusion_matrix(conf_mat=con_mat,
                              colorbar=True,
                              show_absolute=True,
                              show_normed=True,
                              class_names=class_name,
                              figsize=[8, 8])
plt.xlabel('predicted lavel', fontweight = 'bold',fontsize = 10)
plt.ylabel('true lavel', fontweight = 'bold', fontsize = 10)
plt.title('svm',fontweight = 'bold',fontsize = 10)
plt.show()
#%%
##################### GrandBoostClassifier ####################################
from sklearn.ensemble import GradientBoostingClassifier
gbc = GradientBoostingClassifier()
param_gbc = {
    'n_estimators' : [50, 100, 200, 400],
    'max_depth' : [1,2,4,8,16],
    'learning_rate' : [0.01, 0.1, 1, 10, 100]
    }
grid_gbc = GridSearchCV(estimator=gbc, param_grid = param_gbc)
grid_gbc.fit(x_trn, y_trn)
print(grid_gbc.best_params_)
print('GBC train Score',grid_gbc.score(x_trn,y_trn))
print('GBC test Score',grid_gbc.score(x_tst,y_tst))
y_gbc_pred = grid_gbc.predict(x_tst)
y_gbc_prob = grid_gbc.predict_proba(x_tst)
# Plot result
con_mat=confusion_matrix(y_tst,y_gbc_pred)
#print(con_mat)
fig, ax=plot_confusion_matrix(conf_mat=con_mat,
                              colorbar=True,
                              show_absolute=True,
                              show_normed=True,
                              class_names=class_name,
                              figsize=[8, 8])
plt.xlabel('predicted lavel', fontweight = 'bold',fontsize = 10)
plt.ylabel('true lavel', fontweight = 'bold', fontsize = 10)
plt.title('gbc',fontweight = 'bold',fontsize = 10)
plt.show()
#%%
plt.hist(grid_gbc.predict(EXP_070_385),range=(1,7))
plt.show()
#%%
##################### AdaBoostClassfier  ######################################
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
adabst = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(),random_state=42)
param_ada = {
    'n_estimators' : [50, 100, 150, 200],
    'learning_rate' : [0.001, 0.01, 0.1, 1]
    }
grid_abc = GridSearchCV(adabst, param_ada)
grid_abc.fit(x_trn, y_trn)
print('AdaBoost train Score',grid_abc.score(x_trn,y_trn))
print('AdaBoost test Score',grid_abc.score(x_tst,y_tst))
y_abc_pred = grid_abc.predict(x_tst)
y_abc_prob = grid_abc.predict_proba(x_tst)
# Plot result
con_mat=confusion_matrix(y_tst,y_abc_pred)
#print(con_mat)
fig, ax=plot_confusion_matrix(conf_mat=con_mat,
                              colorbar=True,
                              show_absolute=True,
                              show_normed=True,
                              class_names=class_name,
                              figsize=[8, 8])
plt.xlabel('predicted lavel', fontweight = 'bold',fontsize = 10)
plt.ylabel('true lavel', fontweight = 'bold', fontsize = 10)
plt.title('Adaboost',fontweight = 'bold',fontsize = 10)
plt.show()
#%%
############################### knn model #####################################
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
param_knn = {
    'n_neighbors': [3,5,7,9,11,13],
    'weights': ['uniform', 'distance'],
    'p': [1,2]    
    }
grid_knn= GridSearchCV(knn, param_knn, cv = 6)
grid_knn.fit(x_trn,y_trn)
print(grid_knn.best_params_)
print('knn train Score',grid_knn.score(x_trn,y_trn))
print('knn test Score',grid_knn.score(x_tst,y_tst))
y_knn_pred = grid_knn.predict(x_tst)
y_knn_prob = grid_knn.predict_proba(x_tst)
# Plot result
con_mat=confusion_matrix(y_tst,y_knn_pred)
#print(con_mat)
fig, ax=plot_confusion_matrix(conf_mat=con_mat,
                              colorbar=True,
                              show_absolute=True,
                              show_normed=True,
                              class_names=class_name,
                              figsize=[8, 8])
plt.xlabel('predicted lavel', fontweight = 'bold',fontsize = 10)
plt.ylabel('true lavel', fontweight = 'bold', fontsize = 10)
plt.title('knn',fontweight = 'bold',fontsize = 10)
plt.show()
#%%
##################### Stacked Classifier ######################################
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
cls_models = [cls_rf, gbc, adabst, knn]
cls_params = [rf_param_grid, param_gbc, param_ada, param_knn]
grid_rf = GridSearchCV(cls_rf, rf_param_grid).fit(x_trn,y_trn)
grid_svm = GridSearchCV(svm, param_svm).fit(x_trn,y_trn)
grid_gbc = GridSearchCV(gbc, param_gbc).fit(x_trn,y_trn)
grid_abc = GridSearchCV(adabst, param_ada).fit(x_trn,y_trn)
grid_knn = GridSearchCV(knn, param_knn).fit(x_trn,y_trn)
#base models
base_estimator = [
    ('rf', grid_rf.best_estimator_),
    ('adabst', grid_abc.best_estimator_),
    ('gbc', grid_gbc.best_estimator_),
    ('knn', grid_knn.best_estimator_)
    ]
stack_clf = StackingClassifier(
    estimators=base_estimator,stack_method = 'predict_proba',                          
    final_estimator= LogisticRegression()
    )
stack_clf.fit(x_trn, y_trn)
print('stack train Score',stack_clf.score(x_trn,y_trn))
print('stack test Score',stack_clf.score(x_tst,y_tst))
y_pred_stack = stack_clf.predict(x_tst)
y_stack_prob = stack_clf.predict_proba(x_tst)
# Plot confusion matrix
#class_name = ['group 1','group 2','group 3','group 4','group 5','group 6','group 7']
class_name = ['zone 1','zone 2','zone 3','zone 4','zone 5']
con_mat=confusion_matrix(y_tst,y_pred_stack)
#print(con_mat)
fig, ax=plot_confusion_matrix(conf_mat=con_mat,
                              colorbar=False,
                              show_absolute=True,
                              show_normed=True,
                              class_names=class_name,
                              figsize=[6, 6])
plt.xlabel('predicted lavel', fontweight = 'bold',fontsize = 10)
plt.ylabel('true lavel', fontweight = 'bold', fontsize = 10)
plt.title('stack',fontweight = 'bold',fontsize = 10)
plt.show()
#%%
from sklearn.ensemble import VotingClassifier
vote_clf = VotingClassifier(
    estimators=base_estimator,
    voting='hard')
vote_clf.fit(x_trn, y_trn)

print(f'Vote train Score: {vote_clf.score(x_trn,y_trn):.4f}')
print(f'Vote test Score: {vote_clf.score(x_tst,y_tst):.4f}')
y_vote_pred = vote_clf.predict(x_tst)
# Plot confusion matrix
con_mat=confusion_matrix(y_tst,y_vote_pred)
#print(con_mat)
fig, ax=plot_confusion_matrix(conf_mat=con_mat,
                              colorbar=False,
                              show_absolute=True,
                              show_normed=True,
                              class_names=class_name,
                              figsize=[6, 6])
plt.xlabel('predicted lavel', fontweight = 'bold',fontsize = 10)
plt.ylabel('true lavel', fontweight = 'bold', fontsize = 10)
plt.title('vote',fontweight = 'bold',fontsize = 10)
plt.show()
#%%
################### Function For ROC Curve estimetion  ########################
from sklearn.metrics import roc_curve, auc
def ROC_Curve_avg(y_true, y_prob_pred, name, color, n_classes):
    clf_name = name
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true[:,i], y_prob_pred[:,i])
        roc_auc[i] = auc(fpr[i],tpr[i])
    # compute micro-avarage ROC and ROC area
    fpr['micro'], tpr['micro'], _ = roc_curve(y_true.ravel(), y_prob_pred.ravel())
    roc_auc['micro'] = auc(fpr['micro'], tpr['micro'])
    # Plot ROC for all class
    # Average all it and compute AUC
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    # Then interpolate all ROC curve at this point
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    fpr['macro'] = all_fpr
    tpr['macro'] = mean_tpr
    roc_auc['macro'] = auc(fpr['macro'], tpr['macro'])
    # Plot all roc
    area = roc_auc['micro']
    lw = 2
    plt.plot(
        fpr['micro'],
        tpr['micro'],
        #label = 'ROC (area ={0:02f}) '.format( roc_auc['micro']),
        label = f'{clf_name} ROC (area = {"%0.04f"%area})',
        color = color,
        linestyle = '-',
        linewidth = 2,
        )
    plt.plot([0,1],[0,1],'k--', lw= lw)
    plt.xlim(-0.01,1)
    plt.ylim((0,1.05))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc = 'lower right')
#%%
############### ROC For all clf for performance comparision ###################
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(handle_unknown='ignore')
y_true = encoder.fit_transform(y_tst.reshape(-1,1))
import pandas as pd
y_tsts = pd.get_dummies(y_tst)
y_true = np.array(y_tsts)
#%%
clf_name = ['SVM','RF','abc','GBC','knn','stack']
y_prob = [y_svm_prob, y_rf_prob,y_abc_prob, y_gbc_prob, y_knn_prob, y_stack_prob]
color = ['red','green', 'blue','black','orange','maroon']
# Plot ROC
for i in range(6):
    ROC_Curve_avg(
        y_true, 
        y_prob_pred=y_prob[i], 
        name = clf_name[i],
        color=color[i],
        n_classes=7
        )
plt.show()
#%% test with experimetal data
fig, ax = plt.subplots(1,3, figsize=(9,3))
ax[0].hist(stack_clf.predict(EXP_050_265[0:63,:]),range=(1,7))
ax[0].set(xlabel='Size Group', ylabel='Count')
ax[1].hist(stack_clf.predict(EXP_100_265[0:51,:]),range=(1,7))
ax[1].set(xlabel='Size Group')
ax[2].hist(stack_clf.predict(EXP_070_385[50:100,:]),range=(1,7))
ax[2].set(xlabel='Size Group')
plt.show()