import torch
import torch.autograd as autograd
import torch.optim as optim
import torch.utils.data as Data

from evaluator import evaluate
from untils import collate_fn


def train(args, logger, tokenizer, model, train_data, test_data):
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    loader = Data.DataLoader(dataset=train_data,batch_size=args.train_batch_size,shuffle=False,collate_fn=collate_fn)
    best_P = 0
    num_stop_train = args.num_stop_train
    for epoch in range(args.num_train_epochs):
        for step ,(train_x_text,batch_y) in enumerate(loader): 
            mask = torch.ne(train_x_text,0).cuda()
            loss = model(train_x_text, None, mask, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (step + 1 ) % (len(train_data)//args.train_batch_size//2) == 0:
                logger.info("Epoch: {}, Loss: {:2f}".format(epoch + 1, loss.item()))
        P = evaluate(args,tokenizer,model,test_data)
        logger.info('Epoch: {}, F1: {:2f}'.format(epoch + 1,P))
        if P > best_P :
            best_P = P
            num_stop_train = args.num_stop_train
            torch.save(model.state_dict(),"./model.pkl")
            logger.info('Epoch: {}, save the model ,Best_F1: {:2f}'.format(epoch + 1,P))
        else:
            num_stop_train -= 1
            if num_stop_train == 0:
                logger.info("Trainning over, Best_F1: {:2f}",format(best_P))
                break
