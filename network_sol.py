import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

#################### Loss function ####################
def PointNetLoss(x, category, mtx_in, mtx_feature, task, alpha = 0.0001):
    # mtx_in: 3 by 3 / mtx_feature: 64 by 64
    criterion = torch.nn.NLLLoss()
    batch_size = x.size(0)

    idmtx_in = torch.eye(3, requires_grad=True).repeat(batch_size, 1, 1).to(x.device)
    
    if task == 'Classification':
        idmtx_feature = torch.eye(64, requires_grad=False).repeat(batch_size, 1, 1).to(x.device)
    elif task == 'Segmentation':
        idmtx_feature = torch.eye(128, requires_grad=False).repeat(batch_size, 1, 1).to(x.device)

    diff_in = idmtx_in - torch.bmm(mtx_in, mtx_in.transpose(1,2))
    diff_feature = idmtx_feature - torch.bmm(mtx_feature, mtx_feature.transpose(1,2))
    
    loss =  criterion(x, category)
    loss += alpha * (torch.norm(diff_in) + torch.norm(diff_feature)) / float(batch_size)

    return loss

#################### T-Net ####################
class TNet(nn.Module):
   def __init__(self, k=3):
      super().__init__()
      self.k=k
      self.conv1 = nn.Conv1d(self.k, 64, 1)
      self.conv2 = nn.Conv1d(64, 128, 1)
      self.conv3 = nn.Conv1d(128, 1024, 1)

      self.fc1 = nn.Linear(1024, 512)
      self.fc2 = nn.Linear(512, 256)
      self.fc3 = nn.Linear(256, self.k * self.k)

      self.bn1 = nn.BatchNorm1d(64)
      self.bn2 = nn.BatchNorm1d(128)
      self.bn3 = nn.BatchNorm1d(1024)
      self.bn4 = nn.BatchNorm1d(512)
      self.bn5 = nn.BatchNorm1d(256)

   def forward(self, data_in):
        # id matrix
        batch_size = data_in.size(0)
        idmtx = torch.eye(self.k, requires_grad=True).repeat(batch_size, 1, 1).to(data_in.device)

        # forward path
        x = F.relu(self.bn1(self.conv1(data_in)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = nn.MaxPool1d(x.size(-1))(x)
        x = nn.Flatten(1)(x)
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x).view(-1, self.k, self.k)

        return idmtx + x


#################### Transform ####################
class Transform_Classification(nn.Module):
    def __init__(self, use_tnet):
        super().__init__()
        self.use_tnet = use_tnet

        self.input_transform = TNet(3)
        self.feature_transform = TNet(64)

        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
       
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
       
    def forward(self, data_in):
        if self.use_tnet:
            mtx_in = self.input_transform(data_in) # 3 by 3 matrix
        else:
            batch_size = data_in.size(0)
            mtx_in = torch.eye(3, requires_grad=False).repeat(batch_size, 1, 1).to(data_in.device)

        x = torch.bmm(torch.transpose(data_in, 1, 2), mtx_in).transpose(1, 2)

        x = F.relu(self.bn1(self.conv1(x)))

        if self.use_tnet:
            mtx_feature = self.feature_transform(x) # 64 by 64 matrix
        else:
            batch_size = data_in.size(0)
            mtx_feature = torch.eye(64, requires_grad=False).repeat(batch_size, 1, 1).to(data_in.device)

        x = torch.bmm(torch.transpose(x, 1, 2), mtx_feature).transpose(1, 2)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = nn.MaxPool1d(x.size(-1))(x)
        x = nn.Flatten(1)(x)
        
        return x, mtx_in, mtx_feature
    

class Transform_Segmentation(nn.Module):
    def __init__(self, use_tnet):
        super().__init__()
        self.use_tnet = use_tnet

        self.input_transform = TNet(3)
        self.feature_transform = TNet(128)

        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 128, 1)
        self.conv4 = nn.Conv1d(128, 512, 1)
        self.conv5 = nn.Conv1d(512, 2048, 1)
       
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(2048)
       
    def forward(self, data_in):
        out = []

        if self.use_tnet:
            mtx_in = self.input_transform(data_in) # 3 by 3 matrix
        else:
            batch_size = data_in.size(0)
            mtx_in = torch.eye(3, requires_grad=False).repeat(batch_size, 1, 1).to(data_in.device)

        x = torch.bmm(torch.transpose(data_in, 1, 2), mtx_in).transpose(1, 2)

        x = F.relu(self.bn1(self.conv1(x)))
        out.append(x)
        x = F.relu(self.bn2(self.conv2(x)))
        out.append(x)
        x = F.relu(self.bn3(self.conv3(x)))
        out.append(x)

        if self.use_tnet:
            mtx_feature = self.feature_transform(x) # 64 by 64 matrix
        else:
            batch_size = data_in.size(0)
            mtx_feature = torch.eye(128, requires_grad=False).repeat(batch_size, 1, 1).to(data_in.device)

        x = torch.bmm(torch.transpose(x, 1, 2), mtx_feature).transpose(1, 2)
        out.append(x)

        x = F.relu(self.bn4(self.conv4(x)))
        out.append(x)
        x = F.relu(self.bn5(self.conv5(x)))
        x = nn.MaxPool1d(x.size(-1))(x)
        x = nn.Flatten(1)(x)
        
        # for segmentation
        n = data_in.size()[2]
        x = nn.Flatten(1)(x).repeat(n,1,1).transpose(0,2).transpose(0,1)

        out.append(x)
        
        return out, mtx_in, mtx_feature

#################### Classification ####################
class PointNetClassification(nn.Module):
    def __init__(self, class_num = 10, dropout_prob = 0.3, use_tnet = 'yes_t'):
        super().__init__()
        if use_tnet == 'yes_t':
            self.use_tnet = True
        else:
            self.use_tnet = False

        self.transform = Transform_Classification(self.use_tnet)
        self.fc1 = nn.Linear(1024, 512) 
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, class_num)
        
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)

        self.dropout = nn.Dropout(p = dropout_prob)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, data_in):
        x, mtx_in, mtx_feature = self.transform(data_in)
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(self.fc2(x))
        x = F.relu(self.bn2(x))
        x = self.fc3(x)
        x = self.logsoftmax(x)

        return x, mtx_in, mtx_feature

#################### Segmentation ####################
class PointNetSegmentation(nn.Module):
    def __init__(self, classes = 4, use_tnet = 'yes_t'):
        super().__init__()
        if use_tnet == 'yes_t':
            self.use_tnet = True
        else:
            self.use_tnet = False
        

        self.transform = Transform_Segmentation(self.use_tnet)
        self.fc1 = nn.Conv1d(3008, 256, 1) #3008 = 64 + 128x3 + 512 + 2048 # We are only doing one class of part - ignoring the one-hot implementation
        self.fc2 = nn.Conv1d(256, 256, 1) 
        self.fc3 = nn.Conv1d(256, 128, 1) 
        self.fc4 = nn.Conv1d(128, classes, 1)
        
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(classes)
        
        self.logsoftmax = nn.LogSoftmax(dim=1)
        
    def forward(self, data_in):
        x, mtx_in, mtx_feature = self.transform(data_in)
        x = torch.cat(x, 1)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        x = F.relu(self.bn4(self.fc4(x)))
        
        return self.logsoftmax(x), mtx_in, mtx_feature

#################### Get Model ####################
def GetModel(tnet:str, task:str, device:None):
    assert device is not None

    if task == 'Classification':
        model = PointNetClassification(use_tnet = tnet).to(device)
    elif task == 'Segmentation':
        model = PointNetSegmentation(use_tnet = tnet).to(device)
    else:
        return None, None, None

    optimizer = Adam(model.parameters(), lr = 0.001)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.95)

    return model, optimizer, scheduler