import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

#################### Loss function ####################
def PointNetLoss(x, category, mtx_in, mtx_feature, task, alpha = 0.0001):
    # mtx_in: 3 by 3 / mtx_feature: 64 by 64
    #TODO Implement Claissification loss function 
    #Hint: use torch.nn.NLLLoss()

    batch_size = x.size(0)
    criterion = NotImplementedError  

    #TODO Implement regularization loss - equation (2), L_reg in the paper
    # ake sure to send the tensor to the device!
    #NOTE The loss IS task-specific.
    if task == 'Classification':
        pass
    elif task == 'Segmentation':
        pass
    

    L_reg_1 = NotImplementedError # Coming from 3x3 matrix
    L_reg_2 = NotImplementedError # Coming from 64x64 matrix
    
    #TODO Loss = criterion + alpha * (L_reg_1 + L_reg_2)
    loss = NotImplementedError

    return loss

#################### T-Net ####################
class TNet(nn.Module):
   def __init__(self, k=3):
      super().__init__()
      self.k=k

      #TODO Initialize layers 
      self.conv1 = nn.Conv1d(self.k, 64, 1) # You can use the same type of neural network -- this is big hint!
      self.conv2 = NotImplementedError
      self.conv3 = NotImplementedError

      self.fc1 = nn.Linear(1024, 512) 
      self.fc2 = NotImplementedError
      self.fc3 = NotImplementedError

      self.bn1 = nn.BatchNorm1d(64)
      self.bn2 = NotImplementedError
      self.bn3 = NotImplementedError
      self.bn4 = NotImplementedError
      self.bn5 = NotImplementedError

   def forward(self, data_in):
        # id matrix
        batch_size = data_in.size(0)
        idmtx = torch.eye(self.k, requires_grad=True).repeat(batch_size, 1, 1).to(data_in.device)

        # forward path
        x = F.relu(self.bn1(self.conv1(data_in))) #Weight-shared Layer 1
        x = NotImplementedError # Weight-shared Layer 2
        x = NotImplementedError # Weight-shared Layer 3
        x = nn.MaxPool1d(x.size(-1))(x)
        x = nn.Flatten(1)(x)
        x = F.relu(self.bn4(self.fc1(x))) # Fully connected Layer 1
        x = NotImplementedError # Fully connected layer 2
        x = NotImplementedError # Fully connected layer 3
        x = NotImplementedError # reshape into [batchsize,k,k] tensor

        return idmtx + x


#################### Transform ####################
class Transform_Classification(nn.Module):
    def __init__(self, use_tnet):
        super().__init__()
        self.use_tnet = use_tnet # True if we are going to use Tnet (Almost True)

        #TODO
        self.input_transform = NotImplementedError # Output of the Tnet() of size 3
        self.feature_transform = NotImplementedError # Output of the Tnet() of size 64

        #TODO
        self.conv1 = nn.Conv1d(3, 64, 1)  # Weight Shared layer 1 (after input transformation)
        self.conv2 = NotImplementedError # Weight Shared layer 2 (after feature transformation)
        self.conv3 = NotImplementedError # Weight Shared layer 3 (after feature transformation)
       
        #TODO
        self.bn1 = NotImplementedError
        self.bn2 = NotImplementedError
        self.bn3 = NotImplementedError
       
    def forward(self, data_in):
        if self.use_tnet:
            mtx_in = NotImplementedError # 3 by 3 matrix (output of the Tnet)
        else:
            batch_size = data_in.size(0)
            mtx_in = torch.eye(3, requires_grad=False).repeat(batch_size, 1, 1).to(data_in.device)
        
        #TODO Multiply mtx_in to the data_in.
            #HINT: Try using torch.bmm, together with the torch.transpose
            # data_in = [batchsize, 3, num_points]
            # mtx_in = [batchsize, 3, 3]
            # x = [batchsize, 3, num_points]
        x = NotImplementedError

        #Example: Weight shared network after input transformation
        x = F.relu(self.bn1(self.conv1(x)))

        if self.use_tnet:
            mtx_feature = NotImplementedError # 64 by 64 matrix (output of the Tnet)
        else:
            batch_size = data_in.size(0)
            mtx_feature = torch.eye(64, requires_grad=False).repeat(batch_size, 1, 1).to(data_in.device)

        #TODO Feature multiplication part
            # HINT: Try using torch.bmm, together with the torch.transpose
            # data_in = [batchsize, feature_size, num_points]
            # mtx_in = [batchsize, feature_size, feature_size]
            # x = [batchsize, feature_size, num_points]
        x = NotImplementedError

        #TODO Weight shared network 1 after feature transformation
        x = NotImplementedError

        #TODO Weight shared network 2 after feature transformation
        x = NotImplementedError

        #TODO Weight shared network 3 after feature transformation - the activation function for this is maxpool!
        x = NotImplementedError

        # Flattening for input to MLP
        x = nn.Flatten(1)(x)
        
        return x, mtx_in, mtx_feature
    
class Transform_Segmentation(nn.Module):
    def __init__(self, use_tnet):
        super().__init__()
        self.use_tnet = use_tnet

        self.input_transform = TNet(3)
        self.feature_transform = TNet(128)

        #TODO Implementation of the transform network for the segmentation task
        # is different from Fig.2 in the original paper.
        # The actual implementation is Figure 9, page 11 - see the extended version of paper which is uploaded in the arxiv.

        #TODO definition of weight-shared layers
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = NotImplementedError
        self.conv3 = NotImplementedError
        self.conv4 = NotImplementedError
        self.conv5 = NotImplementedError
       
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = NotImplementedError
        self.bn3 = NotImplementedError
        self.bn4 = NotImplementedError
        self.bn5 = NotImplementedError
       
    def forward(self, data_in):
        out = []

        if self.use_tnet:
            mtx_in = NotImplementedError # 3 by 3 matrix (output of the Tnet)
        else:
            batch_size = data_in.size(0)
            mtx_in = torch.eye(3, requires_grad=False).repeat(batch_size, 1, 1).to(data_in.device)

        #TODO Multiply mtx_in to the data_in.
            #HINT: Try using torch.bmm, together with the torch.transpose
            # data_in = [batchsize, 3, num_points]
            # mtx_in = [batchsize, 3, 3]
            # x = [batchsize, 3, num_points]
        x = NotImplementedError

        #TODO Series of weight-shared layers.
        x = NotImplementedError
        out.append(x) # we are taking as the output 
        x = NotImplementedError
        out.append(x) # we are taking as the output 
        x = NotImplementedError
        out.append(x) # we are taking as the output 

        if self.use_tnet:
            mtx_feature = self.feature_transform(x) # 64 by 64 matrix
        else:
            batch_size = data_in.size(0)
            mtx_feature = torch.eye(64, requires_grad=False).repeat(batch_size, 1, 1).to(data_in.device)

        #TODO Multiply mtx_in to the data_in.
            #HINT: Try using torch.bmm, together with the torch.transpose
            # data_in = [batchsize, feature_size, num_points]
            # mtx_in = [batchsize, feature_size, feature_size]
            # x = [batchsize, feature_size, num_points]
        x = NotImplementedError

        out.append(x) # we are taking as the output 

        #TODO Weight-shared layer
        x = NotImplementedError
        out.append(x)
        #TODO Weight-shared layer
        x = NotImplementedError
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

        #TODO
        self.transform = Transform_Classification(self.use_tnet)

        #TODO Definition of the MLP (task-specific head)
        self.fc1 = NotImplementedError
        self.fc2 = NotImplementedError
        self.fc3 = NotImplementedError # the output should be 'class_num'
        
        #TODO Definition of the BN for the MLP
        self.bn1 = NotImplementedError
        self.bn2 = NotImplementedError

        self.dropout = nn.Dropout(p = dropout_prob)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, data_in):
        x, mtx_in, mtx_feature = self.transform(data_in)
        # TODO First Fully connected layer
        x = F.relu(self.bn1(self.fc1(x)))

        # TODO FC2 -> dropout -> batchnorm -> RELU -> FC3 -> Softmax
        x = NotImplementedError

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

        #TODO Definition of the weight shared network
        self.fc1 = nn.Conv1d(3008, 512, 1) #3008 = 64 + 128x3 + 512 + 2048 # We are only doing one class of part - ignoring the one-hot implementation
        self.fc2 = NotImplementedError
        self.fc3 = NotImplementedError 
        self.fc4 = NotImplementedError # output should be classes
        
        self.bn1 = NotImplementedError
        self.bn2 = NotImplementedError
        self.bn3 = NotImplementedError
        self.bn4 = NotImplementedError
        
        self.logsoftmax = nn.LogSoftmax(dim=1)
        
    def forward(self, data_in):
        x, mtx_in, mtx_feature = self.transform(data_in)
        x = torch.cat(x, 1)

        # sequences of weight shared - fully connected networks with ReLU
        x = NotImplementedError #TODO Layer 1
        x = NotImplementedError #TODO Layer 2
        x = NotImplementedError #TODO Layer 3
        x = NotImplementedError #TODO Layer 4
        
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
    scheduler = NotImplementedError # Implement exponentially decaying scheduler with gamma = your selection

    return model, optimizer, scheduler