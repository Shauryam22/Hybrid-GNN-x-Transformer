import torch
from tokenisation import tokenise
def encode(text_list,stoi,block_size=128):
    encoded_data = []
    unk_id = stoi['<unk>']
    pad_id = stoi['<pad>']
    print('Tokenisation is hapenning....')
    for text in text_list:
        tokens = tokenise(text)
        ids = [stoi.get(t,unk_id) for t in tokens]

        if len(ids) > block_size:
            ids = ids[:block_size]
        else:
            ids = ids + [pad_id]*(block_size-len(ids))

        encoded_data.append(ids)
    return torch.tensor(encoded_data,dtype=torch.long)

