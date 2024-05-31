import os.path

import torch
from tqdm import tqdm

from Utilities.Logger.Logger import Logger


class ModelTrainer(Logger):
    """
    This class is made to hide the complexity of and provide a consistent way to train and test models using pytorch
    """

    def __init__(self, model, optimizer, loss_function,checkpoint_folder):
        """
        :param model: model to train
        :param optimizer: optimizer to use
        :param loss_function: loss function to use
        :param checkpoint_folder: where to save the model's checkpoints
        """
        self.model = model
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.checkpoint_folder = checkpoint_folder

    def train(self, training_dataloader, validation_dataloader, epochs):
        """
        Train the model
        :param training_dataloader: dataloader providing data for training
        :param validation_dataloader: dataloader providing data for training
        :param epochs: epochs to use
        :return:
        """

        min_valid_loss = float('inf')

        for epoch in range(epochs):  # loop over the dataset multiple times

            training_loss = 0.0
            self.model.train()
            for i, data in enumerate(tqdm(training_dataloader),start=0):
                # load the data from the given dataloader
                inputs, labels = data

                # load the data in GPU memory if available
                if torch.cuda.is_available():
                    inputs, labels = inputs.cuda(), labels.cuda()

                # set the optimizer's gradient to zero
                self.optimizer.zero_grad()

                # perform the forward pass
                outputs = self.model(inputs)

                # compute the loss
                loss = self.loss_function(outputs, labels)

                # optimize
                loss.backward()
                self.optimizer.step()

                # print statistics
                training_loss += loss.item()

            valid_loss = 0.0
            self.model.eval()  # Optional when not using Model Specific layer
            for data, labels in validation_dataloader:

                # load the data from the given dataloader
                inputs, labels = data

                # load the data in GPU memory if available
                if torch.cuda.is_available():
                    inputs, labels = inputs.cuda(), labels.cuda()

                target = self.model(data)
                loss = self.loss_function(target, labels)
                valid_loss += loss.item()

            self.logger_module.info(f"Epoch {epoch}, trainin_loss:{training_loss}, validation_loss:{valid_loss}")

            if min_valid_loss > valid_loss:
                print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f}) \t Saving The Model')
                min_valid_loss = valid_loss
                # Saving State Dict
                self.model.save(os.path.join(self.checkpoint_folder,f"checkpoint-{epoch}.pth"))
