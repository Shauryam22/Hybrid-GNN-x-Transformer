from sklearn.model_selection import train_test_split
#from load import load_data
from tokenisation import tokenise1,tokenise
from build_vocab import build_vocab1
from encoding import encode
import torch
import pandas as pd
def load_data(device ='cuda'):
    df = pd.read_csv("C:\\Users\\shaurya\\OneDrive\\Desktop\\VS_Code\\gnn_gpt\\AI_Human.csv",encoding='latin')
    #print(df.head())
    df = df.sample(frac=1, random_state=23)
    #df = df.iloc[:,:2]
    #df.dropna(inplace=True)

    train_df,val_df = train_test_split(df,test_size=.2,train_size=.8,random_state=23)
    return train_df,val_df
t,v = load_data()
print(t.head())

def preproc_data(block_size=128,batch_size=64,device='cuda'):
    train_df,val_df = load_data(device)
    #all_tokens = tokenise1(text=" ".join(train_df['text'].tolist()))
    train_df = train_df.dropna(subset=['text'])
    all_tokens = tokenise1(text=" ".join(train_df['text'].tolist()))
    stoi,itos,vocab_size = build_vocab1(all_tokens)
    x_train = encode(train_df['text'],stoi,block_size)
    y_train = torch.tensor(train_df['ai'].values,dtype = torch.long)
    x_val = encode(val_df['text'],stoi,block_size)
    y_val = torch.tensor(val_df['ai'].values,dtype = torch.long)
    from torch.utils.data import TensorDataset, DataLoader
    train_dataset = TensorDataset(x_train,y_train) # we have to do this , so that pytorch 
    # can access and play with the values in x_train tensor

    # train_dataset = (token_ids,y_labels)
    val_dataset = TensorDataset(x_val,y_val)
    #batch_size = 64   # its like taking a chunk of 128 rows together to train 
    # (batch_size,block_size)
    train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
    val_loader = DataLoader(val_dataset,batch_size=batch_size,shuffle=True) #  no shuff
    # making data according to our (batch_size,block_size)

    return vocab_size,itos,train_dataset,val_dataset,train_loader,val_loader