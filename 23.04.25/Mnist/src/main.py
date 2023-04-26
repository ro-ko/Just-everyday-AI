#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os.path
import sys
import fire
import torch
from pathlib import Path

from utils import set_random_seed
from data import Mnist
from models.mymodel.train import MyTrainer
from models.mymodel.eval import MyEvaluator
from utils import log_param
from loguru import logger


def run_mymodel(device, train_data, test_data, hyper_param):
    trainer = MyTrainer(device=device,
                        in_dim=train_data.in_dim,
                        out_dim=train_data.out_dim)

    model = trainer.train_with_hyper_param(train_data=train_data,
                                           hyper_param=hyper_param)

    evaluator = MyEvaluator(device=device)
    accuracy = evaluator.evaluate(model, test_data)

    return accuracy


def main(model='mymodel',
         seed=-1,
         batch_size=100,
         epochs=15,
         learning_rate=0.001):
    """
    Handle user arguments of ml-project-template

    :param model: name of model to be trained and tested
    :param seed: random_seed (if -1, a default seed is used)
    :param batch_size: size of batch
    :param epochs: number of training epochs
    :param learning_rate: learning rate
    """

    logger.add('file.log')
    # Step 0. Initialization
    logger.info("The main procedure has started with the following parameters:")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    set_random_seed(seed=seed, device=device)
    param = dict()
    param['model'] = model
    param['seed'] = seed
    param['device'] = device
    log_param(param)

    # Step 1. Load datasets
    data_path = Path(__file__).parent.parent.absolute().joinpath("datasets")
    train_data = Mnist(data_path=data_path, train=True)
    test_data = Mnist(data_path=data_path, train=False)
    logger.info("The datasets are loaded where their statistics are as follows:")
    logger.info("- # of training instances: {}".format(len(train_data)))
    logger.info("- # of test instances: {}".format(len(test_data)))

    # Step 2. Run (train and evaluate) the specified model

    logger.info("Training the model has begun with the following hyperparameters:")
    hyper_param = dict()
    hyper_param['batch_size'] = batch_size
    hyper_param['epochs'] = epochs
    hyper_param['learning_rate'] = learning_rate
    log_param(hyper_param)

    if model == 'mymodel':
        logger.add('file.log')
        accuracy = run_mymodel(device=device,
                               train_data=train_data,
                               test_data=test_data,
                               hyper_param=hyper_param)

        # - If you want to add other model, then add an 'elif' statement with a new runnable function
        #   such as 'run_my_model' to the below
        # - If models' hyperparamters are varied, need to implement a function loading a configuration file
    else:
        logger.error("The given \"{}\" is not supported...".format(model))
        return

    # Step 3. Report and save the final results
    logger.info("The model has been trained. The test accuracy is {:.4}.".format(accuracy))


if __name__ == "__main__":
    sys.exit(fire.Fire(main))
