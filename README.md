# DA-RNN Project

This repository contains my implementation of a Dual-Stage Attention-Based Recurrent Neural Network (DA-RNN) for time series prediction (https://arxiv.org/abs/1704.02971). This was a personal project for me to learn and experiment with this type of model.

I took a lot of inspiration from KurochkinAlexey's implementation (https://github.com/KurochkinAlexey/DA-RNN), so full credits to him for his excellent work. In my implementation, I added a lot of extra text and comments to explain what is happening at each step.

## Project Structure

The project is organized into several Python files:

- `dataset.py`: Contains functions for preparing the dataset.
- `model.py`: Defines the DA-RNN model architecture and initialization.
- `train.py`: Contains functions for training the model, including the training loop, validation, and early stopping.
- `main.py`: The main script that uses all the above modules to perform the actual training.

## Usage

To run the project, first ensure that you have all the required dependencies installed. Then, simply run the `main.py` script:


This will prepare the dataset, initialize the model, and train it using the provided data. You can adjust the hyperparameters and other settings in the script as needed.

## Acknowledgements

Once again, I would like to thank KurochkinAlexey for his excellent implementation of the DA-RNN model, which served as a great inspiration for this project.

A Dual-Stage Attention-Based Recurrent Neural Network for Time Series Prediction
Qin, Y., Song, D., Cheng, H., Cheng, W., Jiang, G., Cottrell, G.
International Joint Conference on Artificial Intelligence (IJCAI) , 2017

