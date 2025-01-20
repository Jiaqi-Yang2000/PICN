import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
import datetime as datetime
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data
import gc
from openpyxl import Workbook

project_name = 'Test'
EPOCH = 2000
RECORD_TIMES = 800
BATCH_SIZE = 64
LR = 0.001
KERNEL_SIZE = 3
PADDING = int((KERNEL_SIZE-1)/2)
lambda_data = 1
lamda_consistency = 0.2
lamda_thermal = 0.1

Test_points = 2

def select_out(tensor, sensor_location):
    [size,_,row,column] = tensor.shape
    tensor_lu = tensor[:,:,sensor_location['yu'].values,sensor_location['xl'].values]
    tensor_ld = tensor[:,:,sensor_location['yd'].values,sensor_location['xl'].values]
    tensor_ru = tensor[:,:,sensor_location['yu'].values,sensor_location['xr'].values]
    tensor_rd = tensor[:,:,sensor_location['yd'].values,sensor_location['xr'].values]
    return ((tensor_ld+tensor_lu+tensor_rd+tensor_ru)/4.0)

def Normalize(data):
    mx = np.amax(data)
    mn = np.amin(data)
    return 0.9*((data - mn)/(mx - mn))

def Denormalize(data, mx, mn):
    return data*(mx - mn) / 0.9 + mn

### Import Data
os.chdir('C:/Users/Desktop/Data')
file_chdir = os.getcwd()
filecsv_list = []
for root,dirs,files in os.walk(file_chdir):
    for file in files:
        if (os.path.splitext(file)[1] == '.csv' \
        and os.path.splitext(file)[0][0:3]=='res' \
        and datetime.datetime.strptime(os.path.splitext(file)[0][8:], "%Y-%m-%d-%H")>datetime.datetime(2015,6,27) \
        and datetime.datetime.strptime(os.path.splitext(file)[0][8:], "%Y-%m-%d-%H")<datetime.datetime(2019,9,22)):
            filecsv_list.append(file)
radiation_data = []
shape_radiation = (pd.read_csv(filecsv_list[0],header=None).values).shape
radiation_time = pd.date_range(start='6/27/2015', end='9/22/2019', freq='D')

for i in range(len(radiation_time_shifting)):
    temp = 'resample'+radiation_time_shifting[i].strftime('%Y-%m-%d')+'.csv'
    if temp in filecsv_list:
        radiation_data.append([pd.read_csv(temp,header=None).values])
    elif temp not in filecsv_list:
        radiation_data.append([np.zeros((shape_radiation[0], shape_radiation[1]))])

mx_radiation = np.amax(np.array(radiation_data))
mn_radiation = np.amin(np.array(radiation_data))

radiation_original = np.array(radiation_data)
radiation_data = Normalize(radiation_data)
radiation_data = radiation_data.tolist()
print(radiation_original.shape)
radiation = zip(radiation_time, radiation_data)

# import temperature monitoring data
os.chdir('C:/Users/Desktop/Data')
temperature = pd.read_csv('temperature_downsurf_4years.csv', delimiter=',')
measure_time = []
for i in range(len(temperature)):
    measure_time.append(datetime.datetime.strptime(temperature[temperature.columns[0]][i][:], "%Y/%m/%d"))
temperature.drop(columns=temperature.columns[0], inplace=True, axis=1)
temperature['measure_time'] = measure_time
temperature.set_index(['measure_time'], inplace=True)
print(temperature.shape)

## Import enviromentable Variable and so on ..
# environment variables
env_data = pd.read_csv('history weather.csv', delimiter=',').fillna(0)
survey_time = []
for i in range(len(env_data)):
    survey_time.append(datetime.datetime.strptime( \
    env_data['dt_iso'][i][:13], "%Y-%m-%d %H")-datetime.timedelta(hours=10))
env_data.drop(columns=env_data.columns[0:5], inplace=True, axis=1)
env_data['survey_time'] = survey_time
env_data.set_index(['survey_time'], inplace=True)

# air_temperature
air = env_data.loc[radiation_time]['temp'].values
mx_air = np.amax(air)
mn_air = np.amin(air)
air_temper_data = []
for i in range(0,len(radiation_time)):
    air_temp = np.zeros((shape_radiation[0],shape_radiation[1]))+air[i]
    air_temper_data.append(air_temp)

# cloudy
cloudy = env_data.loc[radiation_time]['clouds_all'].values
mx_cloud = np.amax(cloudy)
mn_cloud = np.amin(cloudy)
cloudy_data = []
for i in range(0,len(radiation_time)):
    cloudy_temp = np.zeros((shape_radiation[0],shape_radiation[1]))+cloudy[i]
    cloudy_data.append(cloudy_temp)

# winds
wind = env_data.loc[radiation_time]['wind_speed'].values
wind_data = []
for i in range(0,len(radiation_time)):
    wind_temp = np.zeros((shape_radiation[0],shape_radiation[1]))+wind[i]
    wind_data.append(wind_temp)

# rain
rain = env_data.loc[radiation_time]['rain_1h'].values
rain_data = []
for i in range(0,len(radiation_time)):
    rain_temp = np.zeros((shape_radiation[0],shape_radiation[1]))+rain[i]
    rain_data.append(rain_temp)

air_temper_data = Normalize(air_temper_data)
cloudy_data = Normalize(cloudy_data)
wind_data = Normalize(wind_data)
rain_data = Normalize(rain_data)

# combine environmental data
for i in range(len(radiation_data)):
    radiation_data[i].append(air_temper_data[i])
    radiation_data[i].append(cloudy_data[i])
    radiation_data[i].append(wind_data[i])
    radiation_data[i].append(rain_data[i])

# import object variable which includes the location of sensors
location_sensor_add = pd.read_csv('temperature_downsurf_index.csv', delimiter=',')
location_sensor_add.set_index(['id'])
sensor_location = (location_sensor_add.iloc[:])
temperature_data = temperature.loc[radiation_time][sensor_location['id']]
print(temperature_data.shape)

mx_test_output = np.amax(np.array(temperature_data.values))
mn_test_output = np.amin(np.array(temperature_data.values))

## create CNN network

[all_num, LAYER, ROW, COLUMN] = np.array(radiation_data).shape
print([all_num, LAYER, ROW, COLUMN])
test_num = 0.8*all_num

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels = 1,
                out_channels=2,
                kernel_size=KERNEL_SIZE,
                stride=1,
                padding=PADDING,
            ),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.conv2 = nn.Sequential(
             nn.Conv2d(2, 4, kernel_size=KERNEL_SIZE, stride=1, padding=PADDING),
             nn.Tanh(),
             nn.MaxPool2d(2),
        )
        self.conv3 = nn.Sequential(
             nn.Conv2d(4, 1, kernel_size=KERNEL_SIZE, stride=1, padding=PADDING),
             nn.LeakyReLU(),
             nn.MaxPool2d(2),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(5, 5, 1, stride=1, padding = 0),
            nn.ReLU(),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(5, 1, 1, stride=1, padding = 0),
            nn.ReLU(),
        )

    def forward(self, x1, x2):
        x1 = self.conv1(x1)
        x1 = self.conv2(x1)
        x1 = self.conv3(x1)
        x = torch.cat((x1, x2), dim=1)
        x = self.conv4(x)
        x = self.conv5(x)
        return x

# network setting
cnnw=CNN().cuda()
optimizer=torch.optim.Adam(cnnw.parameters(), lr=LR)
loss_func_MSE = nn.MSELoss()
loss_func = nn.MSELoss()

# input data
for i in range(len(radiation_data)):
    radiation_data[i] = np.array(radiation_data[i]).reshape(LAYER,ROW,COLUMN)
radiation_data = np.array(radiation_data)
test_input = radiation_data[:test_num]
test_output_original = np.array(temperature_data.iloc[:test_num])
test_output = Normalize(np.array(temperature_data.iloc[:test_num]))
test_output = test_output.reshape(test_output.shape[0], 1, test_output.shape[1])

# Select the input points
test_output_select = test_output[:,:,0:Test_points]
print(test_input.shape,test_output.shape)
train_input = radiation_data[test_num-1:]
train_output = np.array(temperature_data.iloc[test_num-1:test_num+len(train_input)])
train_output = train_output.reshape(train_output.shape[0], 1, train_output.shape[1])
print(train_input.shape,train_output.shape)
test_tensor_input = torch.FloatTensor(test_input)
test_tensor_output = torch.FloatTensor(test_output_select)
train_tensor_input = torch.FloatTensor(train_input)
train_tensor_output = torch.FloatTensor(train_output)
train_loader = Data.DataLoader(Data.TensorDataset(train_tensor_input,train_tensor_output) \
              , batch_size=BATCH_SIZE, shuffle=True,drop_last = False)
test_loader = Data.DataLoader(Data.TensorDataset(test_tensor_input,test_tensor_output) \
              , batch_size=BATCH_SIZE, shuffle=True,drop_last = False)
Fill_number = all_num % BATCH_SIZE

# Test data input
input = radiation_data
output = Normalize(np.array(temperature_data.values))
output = output.reshape(output.shape[0], 1, output.shape[1])
tensor_input = torch.FloatTensor(input)
tensor_output = torch.FloatTensor(output)
loader = Data.DataLoader(Data.TensorDataset(tensor_input,tensor_output) \
              , batch_size=BATCH_SIZE, shuffle=False, drop_last = False)

# training
error= []
error_absolute = []
Record_step = 0
Training_time = []
workbook = Workbook()
worksheet = workbook.active
worksheet.append(['Time'])

for epoch in range(EPOCH):
    loss_sum = 0

    for step, (x, y)  in enumerate(train_loader):   # batch data, normalize x when iterate train_loader
        b_x = Variable(x)  # batch x
        b_y = Variable(y)   # batch y
        b_x_conv = b_x[:,0:1,:,:]
        b_x_linear = b_x[:,1:,:,:]
        b_x_denormal_radiation = Denormalize(b_x[:, 0, :, :], mx_radiation, mn_radiation)
        b_x_denormal_cloud = Denormalize(b_x[:,2,:,:], mx_cloud, mn_cloud)

        output = cnnw(b_x_conv, b_x_linear)
        output = select_out(output, sensor_location)
        output= output[:,:,0:Test_points]
        output_denormal_original = Denormalize(output, mx_test_output, mn_test_output)

        ## Loss1: Data

        loss = loss_func(output, b_y)   # cross entropy loss
        loss_sum += lamda_data * loss

        ## Loss2: Thermal

        thermal_loss = 0
        times = 0
        for i in sensor_location['yu'].values:
            j = (sensor_location['xl'].values)[times]
            Miu = 1 - b_x_denormal_cloud[:,2,i,j]  #  cloud reduction coefficient
            thermal_loss += Miu * torch.abs(b_x_denormal_radiation[:,0,i,j])-2.33*torch.abs(
            torch.abs(output_denormal_original[:,0, i, j] - output_denormal_original[:,0, i, j+1]) + torch.abs(output_denormal_original[:,0, i, j] - output_denormal_original[:,0, i, j-1]))
            times = times + 1
        thermal_loss = sum(lamda_thermal * thermal_loss / times / BATCH_SIZE)
        loss_thermal_sum += thermal_loss

        ## Loss3: Consistency

        custom_loss = 0
        for j in range(0, height - 1):
            for k in range(0, width - 1):
                 custom_loss += torch.abs(torch.abs(pred[:, j, k] - pred[:, j + 1, k]) - torch.abs(
                 target[:, j, k] - target[:, j + 1, k])) + torch.abs(
                 torch.abs(pred[:, j, k] - pred[:, j, k + 1]) - torch.abs(target[:, j, k] - target[:, j, k + 1]))
        loss_consistency = lamda_consistency * custom_loss.sum() / (batch_size * width)
        loss_consistency_sum += loss_consistency

    optimizer.zero_grad()  # clear gradients for this training step
    loss_sum.sum().backward()  # backpropagation, compute gradients
    optimizer.step()  # apply gradients

    Batch_Times = all_num / BATCH_SIZE
    loss_sum = loss_sum / Batch_Times
    loss_absolute_sum = loss_absolute_sum / Batch_Times
    loss_consistency_sum = loss_consistency_sum / Batch_Times
    loss_thermal_sum = loss_thermal_sum / Batch_Times

    error_sum.append(loss_sum)
    error_absolute.append(loss_absolute_sum)
    error_consistency.append(loss_consistency_sum)
    error_thermal.append(loss_thermal_sum)

    print('***Epoch:' + str(epoch) + '***')
    print('Error_all:')+ str(error_sum[epoch]))
    print('Error_absolute:' + str(error_absolute[epoch]))
    print('Error_consistency:' + str(error_consistency[epoch]))
    print('Error_thermal:' + str(error_thermal[epoch]))

    Record_times = RECORD_TIMES
    Record_number = EPOCH/Record_times

    if (epoch+1) % Record_number == 0:
        Record_step = Record_step + 1
        Record_str = str(Record_step)
        export_data = []
        temp_field = []
        flag = 0
        for step, (x, y) in enumerate(loader):   # batch data, normalize x when iterate train_loader
            b_x = Variable(x)  # batch x
            b_y = Variable(y)
            b_x_conv = b_x[:,0:1,:,:]
            b_x_linear = b_x[:,1:,:,:]

            output = cnnw(b_x_conv, b_x_linear)
            output_original = output.cpu().detach().numpy()
            output = select_out(output, sensor_location).cpu().detach().numpy()

            output = Denormalize(output, mx_test_output, mn_test_output)
            if flag==1:
                export_data = np.vstack((export_data, output))
                export_data_original = np.vstack((export_data_original, output_original))
            elif flag==0:
                export_data = output
                export_data_original = output_original
                flag = 1
        torch.save(cnnw, './result_network/cnnw_'+project_name+'_['+Record_str+'].pkl')