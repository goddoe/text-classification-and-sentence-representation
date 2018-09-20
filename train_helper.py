from tqdm import tqdm
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F


def evaluate(model, data_iter):
    correct = 0
    total = 0
    loss = 0.

    with torch.no_grad():
        pbar = tqdm(iter(data_iter))
        for batch_i, batch in enumerate(pbar, 1):
            X, X_length = batch.text
            Y_idx = batch.label

            logit = model(X)

            new_loss = F.cross_entropy(logit, Y_idx)

            Y_pred = torch.softmax(logit, dim=1)
            Y_pred_idx = Y_pred.max(dim=1)[1]

            correct += Y_idx.eq(Y_pred_idx).sum().item()
            total += Y_idx.size()[0]

            loss = loss + (new_loss-loss)/batch_i

            pbar.set_description("batch: {}/{}, loss: {:.4f}, accuracy: {:.2f}, {}/{}".format(
                batch_i, len(data_iter), loss, correct/total, correct, total))

    return loss, correct, total


def train(model,
          train_iter,
          test_iter,
          n_epoch=5,
          lr=0.001,
          loss_fn=nn.CrossEntropyLoss,
          optimizer=optim.Adam,
          optimizer_option=None):

    loss_list = []
    loss_fn = loss_fn()

    optim_option = {'lr': lr}
    if optimizer_option:
        optim_option.update(optimizer_option)

    optimizer = optimizer(model.parameters(),
                          **optim_option)

    print("*"*30+"\nTrain Start\n"+"*"*30)
    for epoch_i in range(n_epoch):
        pbar = tqdm(iter(train_iter))
        for batch_i, batch in enumerate(pbar, 1):
            X, X_length = batch.text
            Y = batch.label - 1

            model.zero_grad()
            Y_pred = model(X)
            loss = loss_fn(Y_pred, Y)

            loss_list.append(loss.item())
            pbar.set_description("epoch: {}/{}, batch: {}/{}, loss: {:.4f}".format(
                epoch_i, n_epoch, batch_i, len(train_iter), loss_list[-1]))
            loss.backward()
            optimizer.step()

        (train_loss,
         train_correct,
         train_total) = evaluate(model, train_iter)

        (test_loss,
         test_correct,
         test_total) = evaluate(model, test_iter)
        print("train - loss: {:.4f}, accuracy: {:.2f}, {}/{}".format(
            train_loss, train_correct/train_total, train_correct, train_total))

        print("test - loss: {:.4f}, accuracy: {:.2f}, {}/{}".format(
            test_loss, test_correct/test_total, test_correct, test_total))
        print("-"*30)



