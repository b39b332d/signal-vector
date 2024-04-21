
import torch
import numpy as np
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from torch.utils.tensorboard.writer import SummaryWriter
from datetime import datetime
import torch.nn as nn        

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    device = torch.device('cuda:0')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))

    model = module_arch.load_model()
    
    training_loader = module_data.SignalDataLoader('./data/',128, validation_split=0.05)
    validation_loader = training_loader.split_validation()
    training_loader.preview(model)

    EPOCHS = 200
    epoch_number = 0
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    best_vloss = 1_000_000.
    def train_one_epoch(epoch_index, tb_writer):
        running_loss = 0.
        last_loss = 0.
        for i, data in enumerate(training_loader):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = model.loss(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 10 == 9:
                last_loss = running_loss / 10 # loss per batch
                print('  batch {} loss: {}'.format(i + 1, last_loss))
                tb_x = epoch_index * len(training_loader) + i + 1
                tb_writer.add_scalar('Loss/train', last_loss, tb_x)
                running_loss = 0.

        return last_loss

    for epoch in range(EPOCHS):
        print('EPOCH {}:'.format(epoch_number + 1))

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        avg_loss = train_one_epoch(epoch_number, writer)


        running_vloss = 0.0
        # Set the model to evaluation mode, disabling dropout and using population
        # statistics for batch normalization.
        model.eval()

        # Disable gradient computation and reduce memory consumption.
        with torch.no_grad():
            for i, vdata in enumerate(validation_loader):
                vinputs, vlabels = vdata
                voutputs = model(vinputs)
                vloss = model.loss(voutputs, vlabels)
                running_vloss += vloss
            training_loader.preview(model)
                

        avg_vloss = running_vloss / (i + 1)
        print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

        # Log the running loss averaged per batch
        # for both training and validation
        writer.add_scalars('Training vs. Validation Loss',
                        { 'Training' : avg_loss, 'Validation' : avg_vloss },
                        epoch_number + 1)
        writer.flush()

        # Track best performance, and save the model's state
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_path = './model_save/model_{}_{}_{}'.format(timestamp, epoch_number, avg_vloss)
            torch.save(model.state_dict(), model_path)

        epoch_number += 1


