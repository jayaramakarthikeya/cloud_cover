import time
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from model import UNET
from dataset import CloudDataset
from pathlib import Path
from torch.utils.data import DataLoader
from utils import *

def train(model,train_dl,val_dl,loss_fn,optimizer,acc_fn,epochs=1):
    
    start = time.time()
    model.cuda()
    
    train_loss , val_loss = [] , []
    
    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch,epochs-1))
        print('-'*10)
        
        for phase in ['train','valid']:
            if phase == 'train':
                #training phase
                model.train(True)
                dataloader = train_dl
            else:
                model.train(False)
                dataloader = val_dl
                
            running_loss = 0.0
            running_acc = 0.0
            
            step = 0
            
            for x , y in dataloader:
                if torch.cuda.is_available():
                    x = x.cuda()
                    y = y.cuda()
                step += 1
                
                #forward pass
                if phase == 'train':
                    optimizer.zero_grad()
                    out = model(x)
                    loss = loss_fn(out,y)
                    
                    loss.backward()
                    optimizer.step()
                    
                else:
                    with torch.no_grad():
                        out = model(x)
                        loss = loss_fn(out,y)
                        
                #accuracy
                acc = acc_fn(out,y)
                
                running_acc += acc*dataloader.batch_size
                running_loss += loss*dataloader.batch_size
                
                if step % 10 == 0:
                    print('Current Step : {} , Loss : {} , Acc : {} , AllocMem(Mb) : {} '.format(step,loss,acc,torch.cuda.memory_allocated()/1024/1024))
                    
                
            epoch_loss = running_loss/len(dataloader.dataset)
            epoch_acc = running_acc/len(dataloader.dataset)
            
            print('{} Loss : {:.4f} , Acc : {:.4f}'.format(phase,epoch_loss,epoch_acc))
            
            train_loss.append(epoch_loss) if phase=='train' else val_loss.append(epoch_acc)
            
    time_elapsed = time.time() - start
    print('Training time : {:.0f}m {:.0f}s'.format(time_elapsed//60,time_elapsed%60))
    
    return train_loss , val_loss

if __name__ == "__main__":
    loss_fn = nn.CrossEntropyLoss()
    unet = UNET(4,2)
    base_path = Path('../input/38cloud-cloud-segmentation-in-satellite-images/38-Cloud_training')
    data = CloudDataset(base_path/'train_red',
                    base_path/'train_green',
                    base_path/'train_blue',
                    base_path/'train_nir',
                    base_path/'train_gt')
    train_ds , val_ds = torch.utils.data.random_split(data,(6000,2400))
    print(len(train_ds) , len(val_ds))

    train_dl = DataLoader(train_ds,batch_size=2,shuffle=True,pin_memory=True,num_workers=3)
    val_dl = DataLoader(val_ds,batch_size=2,pin_memory=True,num_workers=3)
    xb , yb = next(iter(train_dl))
    opt = torch.optim.Adam(unet.parameters(),lr=0.01)
    train_loss , val_loss = train(unet,train_dl,val_dl,loss_fn,opt,acc_metric,epochs=50)

    plt.plot(train_loss,label='train_loss')
    plt.plot(val_loss,label='val_loss')
    plt.legend()
    plt.show()

    xb, yb = next(iter(train_dl))

    with torch.no_grad():
        predb = unet(xb.cuda())

    predb.shape

    bs = 2
    fig, ax = plt.subplots(bs,3, figsize=(15,bs*5))
    for i in range(bs):
        ax[i,0].imshow(batch_to_img(xb,i))
        ax[i,1].imshow(yb[i])
        ax[i,2].imshow(predb_to_mask(predb, i))

    