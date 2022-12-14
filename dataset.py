import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


from torch.utils.data import Dataset , DataLoader , sampler
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import torch

class CloudDataset(Dataset):
    def __init__(self,r_dir,g_dir,b_dir,nir_dir,gt_dir,pytorch=True):
        super().__init__()
        self.pytorch = pytorch
        
        #loop through the files in the red folder and combine , into a dictionary , the other bands
        self.files = [self.combine_files(f,g_dir,b_dir,nir_dir,gt_dir) for f in r_dir.iterdir() if not f.is_dir()]
        
    def combine_files(self,r_file:Path,g_dir,b_dir,nir_dir,gt_dir):
        
        files = {'red':r_file,
                'green':g_dir/r_file.name.replace('red','green'),
                'blue':b_dir/r_file.name.replace('red','blue'),
                'nir':nir_dir/r_file.name.replace('red','nir'),
                'gt':gt_dir/r_file.name.replace('red','gt')}
        
        return files
    
    def open_as_array(self,idx,invert=False,add_nir=False):
        
        raw_rgb = np.stack([np.array(Image.open(self.files[idx]['red'])),
                           np.array(Image.open(self.files[idx]['green'])),
                           np.array(Image.open(self.files[idx]['blue']))],axis=2)
        
        if add_nir:
            nir = np.expand_dims(np.array(Image.open(self.files[idx]['nir'])),2)
            raw_rgb = np.concatenate([raw_rgb,nir],axis=2)
            
        if invert:
            raw_rgb = raw_rgb.transpose((2,0,1))
            
        #normalize
        return (raw_rgb/np.iinfo(raw_rgb.dtype).max)
    
    def open_mask(self,idx,add_dims=False):
        raw_mask = np.array(Image.open(self.files[idx]['gt']))
        raw_mask = np.where(raw_mask==255,1,0)
        
        return np.expand_dims(raw_mask,0) if add_dims else raw_mask
    
    def __getitem__(self,idx):
        
        x = torch.tensor(self.open_as_array(idx,invert=self.pytorch,add_nir=True),dtype=torch.float32)
        y = torch.tensor(self.open_mask(idx),dtype=torch.int64)
        
        return x , y
    
    def open_as_pil(self,idx):
        
        arr = 256*self.open_as_array(idx)
        
        return Image.fromarray(arr.astype(np.uint8),'RGB')
    
    def __repr__(self):
        s = 'Dataset contains {} files'.format(self.__len__())
        return s
    
    def __len__(self):
        return len(self.files)

if __name__ == "__main__":
    base_path = Path('../input/38cloud-cloud-segmentation-in-satellite-images/38-Cloud_training')
    data = CloudDataset(base_path/'train_red',
                    base_path/'train_green',
                    base_path/'train_blue',
                    base_path/'train_nir',
                    base_path/'train_gt')
    len(data)

    x , y = data[1000]

    fig , ax = plt.subplots(1,2,figsize=(10,9))
    ax[0].imshow(data.open_as_array(26))
    ax[1].imshow(data.open_mask(26))

    train_ds , val_ds = torch.utils.data.random_split(data,(6000,2400))
    print(len(train_ds) , len(val_ds))

    train_dl = DataLoader(train_ds,batch_size=2,shuffle=True,pin_memory=True,num_workers=3)
    val_dl = DataLoader(val_ds,batch_size=2,pin_memory=True,num_workers=3)

    xb , yb = next(iter(train_dl))
    print(xb.shape , yb.shape)