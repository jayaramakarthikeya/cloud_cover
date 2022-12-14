import torch

def get_device():
    return (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))

def to_device(data,device):
    if isinstance(data,(list,tuple)):
        return [to_device(x,device) for x in data]
    return data.to(device)

class DeviceDataLoader:
    
    def __init__(self,dl,device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        for x in self.dl:
            yield to_device(x,self.device)
            
    def __len__(self):
        return len(self.dl)

def batch_to_img(xb, idx):
    img = np.array(xb[idx,0:3])
    return img.transpose((1,2,0))

def predb_to_mask(predb, idx):
    p = torch.functional.F.softmax(predb[idx], 0)
    return p.argmax(0).cpu()

def acc_metric(predb,yb):
    return (predb.argmax(dim=1) == yb.cuda()).float().mean()