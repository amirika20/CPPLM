# Directory configuration
import os
import sys
sys.path.append("/home/amirka/CPP/CPPLM")
sys.path.append("/home/amirka/CPP/CPPLM/project/CPPLM1_1")

# Model related import
from model.model import CPPLM
from training.evaluate import *

# Training related import
from training.dataloader import CPPDataloader
from training.config import TrainingConfig


# Package import
import torch.optim as optim
import pandas as pd
from tqdm import tqdm
import pickle
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import tempfile
from pathlib import Path
import ray
from ray import train
from ray.train import Checkpoint, get_checkpoint

# Constants import
from tokenization.constants import *
from training.viz import viz


def mask(batch, padding_mask, mask_token=SEQUENCE_MASK_TOKEN, mask_prob=0.15, device="cuda"):
    batch_size, max_length = batch.size()
    random_mask = (torch.rand(batch_size, max_length) < mask_prob).to(device)
    final_mask = random_mask & padding_mask.bool()
    masked_batch = batch.clone()
    masked_batch[final_mask] = mask_token
    masked_positions = final_mask.int()
    return masked_batch, masked_positions


class CPPLMTrainer():

    def __init__(
            self,
            model,
    ):
        # Model
        self.model = model
        self.model_name = model.name

        # Optimizer data
        self.optimizer = None
        self.scheduler = None
        self.epoch = 0
        self.best_val_loss = float("inf")
        self.training_losses = []
        self.val_losses = []
        self.test_metrics = None
        self.lr = None
        self.weight_decay = None
    
    def configure_optimizer(
            self,
            optimizer= optim.Adam,
            lr: float= 0.01,
            weight_decay: float= 0.001,
            scheduler= optim.lr_scheduler.ExponentialLR,
            gamma: float= 1,
    ):
        self.optimizer = optimizer(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.75, patience=5)
        self.lr = lr


    def load_checkpoint(self, dir_path="/home/amirka/CPP/CPPLM/project/CPPLM1_1/parameters"):
        path = dir_path + f'/{self.model_name}.pt'
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']
        self.training_losses = checkpoint['training_losses']
        self.val_losses = checkpoint['val_losses']
        self.test_metrics = checkpoint['test_metrics']


    def save_checkpoint(self, dir_path="/home/amirka/CPP/CPPLM/project/CPPLM1_1/parameters"):
        path = dir_path + f'/{self.model_name}.pt'
        checkpoint = {
            'epoch':self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_losses': self.training_losses,
            'val_losses': self.val_losses,
            'test_metrics': self.test_metrics
        }
        torch.save(checkpoint, path)


    def calc_loss(self, intensity_prediction, intensity_target):
        criterion = nn.CrossEntropyLoss()
        loss = criterion(intensity_prediction, intensity_target)
        return loss


    def validation_step(self, batch):
        cpps_ids, padding_mask, intensities, intensity_ids = batch
        cpps_ids, padding_mask, intensities, intensity_ids = cpps_ids.to("cuda"), padding_mask.to("cuda"), intensities.to("cuda"), intensity_ids.to("cuda")
        output = self.model(cpps_ids, padding_mask)
        loss = self.calc_loss(output.predicted_bin, intensity_ids)       
        return loss
    
    def validation(self, val_loader, epoch, num_epochs):
        self.model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f'Validation Epoch {epoch+1}/{num_epochs}'):
                loss = self.validation_step(batch)
                val_loss += loss
        avg_val_loss = val_loss / len(val_loader)

        if avg_val_loss < self.best_val_loss:
            self.best_val_loss = avg_val_loss
            self.save_checkpoint()

        return avg_val_loss
    
    def testing_step(self, batch):
        cpps_ids, padding_mask, intensities, intensity_ids = batch
        cpps_ids, padding_mask, intensities, intensity_ids = cpps_ids.to("cuda"), padding_mask.to("cuda"), intensities.to("cuda"), intensity_ids.to("cuda")
        output = self.model(cpps_ids, padding_mask)
        loss = self.calc_loss(output.predicted_bin, intensity_ids) 
        metrics = self.evaluate(output.predicted_bin, intensity_ids) #TODO
        return loss, metrics
    
    def test(self, test_loader):
        self.load_checkpoint()
        self.model.eval()
        test_loss = 0
        with torch.no_grad():
            for batch in tqdm(test_loader, desc='Testing'):
                loss, metrics = self.testing_step(batch)
                test_loss += loss
        return test_loss / len(test_loader), metrics

    def training_step(self, batch):

        self.optimizer.zero_grad()

        cpps_ids, padding_mask, intensities, intensity_ids = batch
        cpps_ids, padding_mask, intensities, intensity_ids = cpps_ids.to("cuda"), padding_mask.to("cuda"), intensities.to("cuda"), intensity_ids.to("cuda")
        output = self.model(cpps_ids, padding_mask)
        loss = self.calc_loss(output.predicted_bin, intensity_ids)       
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        return loss
    
    def training_epoch(self, train_loader, epoch, num_epochs):
        epoch_loss = 0
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            loss = self.training_step(batch)
            epoch_loss += loss
        avg_epoch_loss = epoch_loss/len(train_loader)
        return avg_epoch_loss
    
    def train(self, train_loader, val_loader, test_loader, training_config):
        epochs_to_go = training_config.num_epochs
        self.configure_optimizer(lr=training_config.lr, weight_decay=training_config.weight_decay)
        for epoch in range(epochs_to_go):
            self.model.train()
            training_loss = self.training_epoch(train_loader, epoch, epochs_to_go)
            val_loss = self.validation(val_loader, epoch, epochs_to_go)
            print(f"[Epoch {epoch + 1}] training loss: {training_loss}")
            print(f"[Epoch {epoch + 1}] validation loss: {val_loss}")
            self.training_losses.append(training_loss.cpu().detach().numpy())
            self.val_losses.append(val_loss.cpu().detach().numpy())
            self.scheduler.step(val_loss)
            self.epoch += 1
        test_loss, metrics = self.test(test_loader)
        print(f"Test loss: {test_loss:.3f}")
        # self.plot_losses()
        print(metrics)
        return metrics



    @staticmethod
    def evaluate(predicted, observed):
        return Evaluation.classification_evaluation(predicted, observed.squeeze())

    def viz(self, datapoint):
        cpps_ids, padding_mask, intensities, intensity_ids = datapoint
        cpps_ids, padding_mask, intensities, intensity_ids = cpps_ids.to("cuda"), padding_mask.to("cuda"), intensities.to("cuda"), intensity_ids.to("cuda")
        output = self.model(cpps_ids, padding_mask)
        loss = self.calc_loss(output.predicted_intensity, intensities) 
        plt.imshow(output.embedding.cpu().detach().numpy(), cmap='viridis')
        plt.colorbar()


    def plot_losses(self):

        epochs = range(1, len(self.training_losses) + 1)
    
        plt.figure(figsize=(10, 5))
        
        plt.plot(epochs, self.training_losses, label='Training Loss')
        plt.plot(epochs, self.val_losses, label='Validation Loss')
        
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss over Epochs')
        plt.legend()
        
        plt.show()


if __name__ == "__main__":
    dataset = CPPDataloader()
    train_loader, val_loader, test_loader = dataset.prep_data()

    # # Building Model
    model = CPPLM(256, 2, 2).to("cuda")

    # # Building Trainder
    training_config = TrainingConfig(lr=0.001,num_epochs=300, weight_decay=1)
    trainer = CPPLMTrainer(model)

    trainer.train(train_loader, val_loader, test_loader, training_config)
    # viz(model)

    # results = []
    # for d_model in [256, 512]:
    #     for n_heads in [1, 2, 4, 8]:
    #         for n_layers in [1, 2, 4, 8]:
    #             for lr in [0.00005]:
    #                 model = CPPLM(d_model, n_heads, n_layers).to("cuda")
    #                 training_config = TrainingConfig(lr=lr, num_epochs=40, weight_decay=0.005)
    #                 trainer = CPPLMTrainer(model)
    #                 # trainer.configure_optimizer(lr=lr)
    #                 metrics = trainer.train(train_loader, val_loader, test_loader, training_config)
    #                 print(metrics)
    #                 results.append({
    #                     "d_model":d_model,
    #                     "n_heads":n_heads,
    #                     "n_layers":n_layers,
    #                     "accuracy":metrics.accuracy,
    #                     "precision":metrics.precision,
    #                     "f1":metrics.f1,
    #                     "recall":metrics.recall,
    #                     "spearman":metrics.spearman
    #                 })

    # df = pd.DataFrame(results)
    # sorted_df = df.sort_values(by='accuracy', ascending=False)
    # sorted_df.to_csv("/home/amirka/CPP/CPPLM/project/CPPLM1_1/results_4_9_13.csv")

    # results = []
    # for d_model in [128, 256, 512, 1024]:
    #     for n_heads in [1, 2, 4, 8]:
    #         for n_layers in [1, 2, 4, 8]:
    #             for lr in [0.00005]:
    #                 model = CPPLM(d_model, n_heads, n_layers).to("cuda")
    #                 training_config = TrainingConfig(lr=0.00005, num_epochs=40, weight_decay=0.005)
    #                 trainer = CPPLMTrainer(model)
    #                 # trainer.configure_optimizer(lr=lr)
    #                 metrics = trainer.train(train_loader, val_loader, test_loader, training_config)
    #                 print(metrics)
    #                 results.append({
    #                     "d_model":d_model,
    #                     "n_heads":n_heads,
    #                     "n_layers":n_layers,
    #                     "lr":lr,
    #                     "uRMSE":metrics.uRMSE,
    #                     "RMSE":metrics.RMSE,
    #                     "uMSE":metrics.uMSE,
    #                     "R2":metrics.R2,
    #                     "r":metrics.r,
    #                     "rho":metrics.rho,
    #                 })

    # df = pd.DataFrame(results)
    # sorted_df = df.sort_values(by='RMSE')
    # sorted_df.to_csv("results.csv")


    