# agreement methods in jaccard distance:

import numpy as np
from tqdm import tqdm

import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

def predict_proba(X_docs):
    X_vect = torch.tensor(tf_vec.transform(X_docs).todense(), dtype=torch.float32)
    preds = model(X_vect).detach().numpy()
    prob = np.concatenate((1-preds, preds), axis=1)
    return prob

def top_features(e1_feat, e2_feat):
    # all features jaccard agreement
    if len(np.union1d(e1_feat, e2_feat)) > 0:
        return len(np.intersect1d(e1_feat, e2_feat)) / len(np.union1d(e1_feat, e2_feat))
    else:
        return 0


def pos_features_agg(
    e1_feat, e1_weight, e2_feat, e2_weight, thresh=0, return_sum=False
):
    pos1 = e1_feat[np.where(e1_weight > thresh)[0]]
    pos2 = e2_feat[np.where(e2_weight > thresh)[0]]
    if return_sum:
        # return feature intersection as well as feature total
        return (
            top_features(pos1, pos2),
            e1_weight[np.where(e1_weight > thresh)[0]].sum(),
            e2_weight[np.where(e2_weight > thresh)[0]].sum(),
        )
    else:
        return top_features(pos1, pos2)


def neg_features_agg(
    e1_feat, e1_weight, e2_feat, e2_weight, thresh=0, return_sum=False
):
    neg1 = e1_feat[np.where(e1_weight < thresh)[0]]
    neg2 = e2_feat[np.where(e2_weight < thresh)[0]]
    if return_sum:
        # return feature intersection as well as feature total
        return (
            top_features(neg1, neg2),
            e1_weight[np.where(e1_weight < thresh)[0]].sum(),
            e2_weight[np.where(e2_weight < thresh)[0]].sum(),
        )
    else:
        return top_features(neg1, neg2)


def pred_features_agg(e1_feat, e1_weight, y1, e2_feat, e2_weight, y2):
    if y1 == 1:
        pred1 = e1_feat[np.where(e1_weight > 0)[0]]
    else:
        pred1 = e1_feat[np.where(e1_weight < 0)[0]]
    if y2 == 1:
        pred2 = e2_feat[np.where(e2_weight > 0)[0]]
    else:
        pred2 = e2_feat[np.where(e2_weight < 0)[0]]
    return top_features(pred1, pred2)

# pytorch
class Wide(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.hidden = nn.Linear(input_size, 10)
        self.relu = nn.ReLU()
        self.output = nn.Linear(10, 1)
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, x):
        x = self.relu(self.hidden(x))
        x = self.sigmoid(self.output(x))
        return x
    
class ExpDataset(Dataset):
    """EXP dataset for pytorch"""

    def __init__(self, X_train, y_train, weights=[], ref_pred=[]):
        self.X_train = X_train
        self.y_train = y_train
        self.weights = weights
        self.ref_pred = ref_pred
        

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
              
        if len(self.weights) > 0: 
            sample = {"x": self.X_train[idx], 
                      "y": self.y_train[idx], 
                      "w": self.weights[idx]}
        elif len(self.ref_pred) > 0: 
            sample = {"x": self.X_train[idx], 
                      "y": self.y_train[idx], 
                      "ref": self.ref_pred[idx]}
        else: 
            sample = {"x": self.X_train[idx], 
                      "y": self.y_train[idx]}

        return sample

    
def model_train_base(input_model, dataloader, X_val, y_val, n_epochs=50, save_best=True):
    # loss function and optimizer

    optimizer = optim.Adam(input_model.parameters(), lr=0.001)
 
    # Hold the best model
    best_acc = - np.inf   # init to negative infinity
    best_weights = None
    best_disagree = 0 
 
    loss_fn = nn.BCELoss() 

    for epoch in tqdm(range(n_epochs)):
        for batch in dataloader: 
              # forward pass
            y_pred = input_model(batch['x'])
            loss = loss_fn(y_pred, batch['y'])
            # backward pass
            optimizer.zero_grad()
            loss.backward()
            # update weights
            optimizer.step()
            # print progress
            
        acc = (y_pred.round() == batch['y']).float().mean()
        # evaluate accuracy at end of each epoch
        input_model.eval()
        y_pred = input_model(X_val)
        acc = (y_pred.round() == y_val).float().mean()
        acc = float(acc)


        if acc > best_acc:
            best_acc = acc
            best_weights = copy.deepcopy(input_model.state_dict())
    # restore model and return best accuracy
    input_model.load_state_dict(best_weights)
    best_acc = acc
    return best_acc

def model_train_tgt(input_model, dataloader, X_val, y_val, instance, pred, n_epochs=50):
    # loss function and optimizer

    optimizer = optim.Adam(input_model.parameters(), lr=0.001)
 
    # Hold the best model
    best_acc = - np.inf   # init to negative infinity
    best_weights = None
    best_disagree = 0 
 
    loss_fn = nn.BCELoss() 

    for epoch in range(n_epochs):
        for batch in dataloader: 
              # forward pass
            y_pred = input_model(batch['x'])
            loss = loss_fn(y_pred, batch['y'])
            # backward pass
            optimizer.zero_grad()
            loss.backward()
            # update weights
            optimizer.step()
            # print progress

        acc = (y_pred.round() == batch['y']).float().mean()
        # evaluate accuracy at end of each epoch
        input_model.eval()
        y_pred = input_model(X_val)
        acc = (y_pred.round() == y_val).float().mean()
        acc = float(acc)
        success = input_model(instance).round() != pred
        if success: 
            print(f"Success! Epoch {epoch}")
        return acc

        if acc > best_acc:
            best_acc = acc
            best_weights = copy.deepcopy(input_model.state_dict())
    # restore model and return best accuracy
    input_model.load_state_dict(best_weights)
    best_acc = acc
    return best_acc

def model_train_reg(model, dataset, X_val, y_val, ref_model, n_epochs=100, batch_size=20, c=0):
    # loss function and optimizer
    loss_fn = nn.BCELoss()  # binary cross entropy
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
 
    # Hold the best model
    best_acc = - np.inf   # init to negative infinity
    best_weights = None
    best_disagree = 0 
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in tqdm(range(n_epochs)):
        for batch in dataloader: 
              # forward pass
            y_pred = model(batch['x'])
            match_loss = c*loss_fn(y_pred, 1 - batch['ref'])
            loss = loss_fn(y_pred, batch['y']) + match_loss
            
            # backward pass
            optimizer.zero_grad()
            loss.backward()
            # update weights
            optimizer.step()
            # print progress
        
        model.eval()
        with torch.no_grad(): 
            yhat1 = ref_model(X_val.detach().numpy())
            yhat2 = model(X_val).round().detach().numpy().flatten()
            disagreement = np.where(yhat1 != yhat2)[0]
            dis_ratio_val = len(disagreement)/ len(yhat1)

            # evaluate accuracy at end of each epoch
            y_pred = model(X_val)
            acc = (y_pred.round() == y_val).float().mean()
            acc = float(acc)

            if acc > best_acc:
                best_acc = acc
                best_weights = copy.deepcopy(model.state_dict())
                best_disagree = dis_ratio_val
                best_dis_weights = copy.deepcopy(model.state_dict())

            if best_acc - acc < 0.05 and dis_ratio_val > best_disagree:
                best_disagree = dis_ratio_val
                best_dis_weights = copy.deepcopy(model.state_dict())
                
    # restore model and return best accuracy
    model.load_state_dict(best_dis_weights)
    return acc


def model_train_weight(model, dataset, X_val, y_val, ref_model, n_epochs=100, batch_size=20, c=0):
    #data loader 
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    #optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)
 
    # Hold the best model
    best_acc = - np.inf   # init to negative infinity
    best_weights = None
    best_disagree = 0 

    model.train()
    for epoch in tqdm(range(n_epochs)):
        for batch in dataloader: 
              # forward pass
            y_pred = model(batch['x'])
            loss_fn = nn.BCELoss(weight=1 + c*batch['w'])
            
            loss = loss_fn(y_pred, batch['y'])
            # backward pass
            optimizer.zero_grad()
            loss.backward()
            # update weights
            optimizer.step()
            # print progress
        
        model.eval()
        with torch.no_grad(): 
            yhat1 = ref_model(X_val.detach().numpy())
            yhat2 = model(X_val).round().detach().numpy().flatten()
            disagreement = np.where(yhat1 != yhat2)[0]
            dis_ratio_val = len(disagreement)/ len(yhat1)

            # evaluate accuracy at end of each epoch
            y_pred = model(X_val)
            acc = (y_pred.round() == y_val).float().mean()
            acc = float(acc)

            if acc > best_acc:
                best_acc = acc
                best_weights = copy.deepcopy(model.state_dict())
                best_disagree = dis_ratio_val
                best_dis_weights = copy.deepcopy(model.state_dict())

            if best_acc - acc < 0.05 and dis_ratio_val > best_disagree:
                #print(acc, best_acc, dis_ratio_val, best_disagree)
                best_disagree = dis_ratio_val
                best_dis_weights = copy.deepcopy(model.state_dict())
    # restore model and return best accuracy
    model.load_state_dict(best_dis_weights)
    return acc