import numpy as np
import matplotlib.pyplot as plt
import sys

import config
"""
Python script to plot the loss curve
"""
def plot_losses(mse, epochs):
    plt.plot(mse)
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.axis([0, epochs, 0, np.max(mse)+0.1])
    plt.title('Loss Curve')
    plt.savefig(config.model_path +'/training_loss_'+str(epochs)+'.png')
    plt.clf()
