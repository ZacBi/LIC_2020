import torch


def Myprocess(indexed_tokens):
    input_ids = []
    lengths = [len(x) for x in indexed_tokens]
    mlen = max(lengths)
    for ids in indexed_tokens:
        l = len(ids)
        ids = ids + [0]*(mlen-l)
        input_ids.append(ids)
    return torch.tensor(input_ids).cuda(), lengths

def collate_fn(batch):
    xs = [v[0] for v in batch]
    ys = [v[1] for v in batch]
    x_ids, xlens = Myprocess(xs)
    return x_ids, torch.tensor(ys).cuda() 
