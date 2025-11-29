import matplotlib.pyplot as plt
import os
import numpy as np

def plot_acc(accuracies, test_type, labels, n_clients, title):
    os.makedirs(f'plots/{test_type}', exist_ok=True)  # creates 'plots/{test_type}' if it doesn't exist to separate depending on what test we are using
    # Plot the accuracies
    plt.figure(figsize=(10, 6))

    for framework_idx, framework_acc in enumerate(accuracies):
        plt.plot(framework_acc, label = f"{labels[framework_idx]} / final acc ~ {framework_acc[-1] * 100:0.2f}%")
    
    # Add labels and title
    plt.xlabel('Communication Rounds')
    plt.ylabel('Accuracy')
    plt.title(f"{title} ({n_clients} clients total)")
    plt.legend()
    plt.grid(True)

    plt.savefig(f'plots/{test_type}/test_accuracy_{test_type}_{n_clients}clnts.png')

def plot_loss(losses, test_type, labels, n_clients, title):
    os.makedirs(f'plots/{test_type}', exist_ok=True)  # creates 'plots/{test_type}' if it doesn't exist to separate depending on what test we are using
    # Plot the losses
    plt.figure(figsize=(10, 6))
    
    for framework_idx, framework_loss in enumerate(losses):
        plt.plot(framework_loss, label = f"{labels[framework_idx]} / final loss ~ {framework_loss[-1]:0.3f}")
    
    # Add labels and title
    plt.xlabel('Communication Round Time (in sec)')
    plt.ylabel('Loss')
    plt.title(f"{title} ({n_clients} clients total)")
    plt.legend()
    plt.grid(True)

    plt.savefig(f'plots/{test_type}/test_loss_{test_type}_{n_clients}clnts.png')


def plot_comm_times(comm_times, test_type, labels, n_clients, title):
    os.makedirs(f'plots/{test_type}', exist_ok=True)  # creates 'plots/{test_type}' if it doesn't exist to separate depending on what test we are using
    # Plot the losses
    plt.figure(figsize=(10, 6))
    
    for framework_idx, framework_comm_times in enumerate(comm_times):
        plt.plot(framework_comm_times, label = f"{labels[framework_idx]} / mean time = {np.mean(framework_comm_times):0.2f}")
    
    # Add labels and title
    plt.xlabel('Communication Rounds')
    plt.ylabel('Communication Times')
    plt.title(f"{title} ({n_clients} clients total)")
    plt.legend()
    plt.grid(True)

    plt.savefig(f'plots/{test_type}/test_comm_times_{test_type}_{n_clients}clnts.png')

