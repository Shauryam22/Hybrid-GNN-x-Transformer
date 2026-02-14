#putting everything in a function.
from collections import Counter
def build_vocab(tokens,max_vocab=10000):
    counts = Counter(tokens)
    items = sorted(counts.items(),key=lambda x: x[1],reverse=True)
    items = items[:max_vocab]
    stoi = {tok:i+1 for i,(tok,_) in enumerate(items)}
    stoi['<pad>'] = 0
    if '<unk>' not in stoi:
        stoi['<unk>'] = len(stoi)
    itos = {i:tok for tok,i in stoi.items()}
    return stoi,itos,len(stoi)