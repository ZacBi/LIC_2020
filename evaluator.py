import numpy as np
import torch.autograd as autograd
import torch.utils.data as Data
from sklearn.metrics import f1_score, roc_auc_score

from untils import *


def evaluate(args,tokenizer,model,datasets):
    with torch.no_grad():
        loaders = Data.DataLoader(dataset=datasets,batch_size=args.test_batch_size,shuffle=False,collate_fn=collate_fn)
        predict = torch.tensor([]).cuda().long()
        target = torch.tensor([]).cuda().long()
        for step ,(train_x_text,batch_y) in enumerate(loaders):  
            mask = torch.ne(train_x_text,0).cuda()
            logits = model(train_x_text, None, mask)
            logits_score,pred = torch.max(logits, dim=-1)
            predict = torch.cat([predict,pred])
            target = torch.cat([target,batch_y])

    return f1_score(target.cpu(),predict.cpu(),average='micro')
