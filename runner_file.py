import argparse
from argparse import RawTextHelpFormatter
import torch.nn as nn
import torch.optim as optim
import random 
import matplotlib.pyplot as plt
import sys
import os

from models import ConvNet
from data_utils import build_mnist
from train_utils import fl_semidecentralized_cluster_train, fl_centralized_train, fl_fullydecentralized_cluster_train



def plot_acc(accuracies, test_type, labels, n_clients, title):
    os.makedirs('plots', exist_ok=True)  # creates 'plots/' if it doesn't exist
    # Plot the accuracies
    plt.figure(figsize=(10, 6))

    for framework_idx, framework_acc in enumerate(accuracies):
        plt.plot(framework_acc, label = f"{labels[framework_idx]}")
    
    # Add labels and title
    plt.xlabel('Communication Rounds')
    plt.ylabel('Accuracy')
    plt.title(f"{title} ({n_clients} clients total)")
    plt.legend()
    plt.grid(True)

    plt.savefig(f'plots/test_accuracy_{test_type}_{n_clients}clnts.png')

def plot_loss(losses, test_type, labels, n_clients, title):
    os.makedirs('plots', exist_ok=True)  # creates 'plots/' if it doesn't exist
    # Plot the losses
    plt.figure(figsize=(10, 6))
    
    for framework_idx, framework_loss in enumerate(losses):
        plt.plot(framework_loss, label = f"{labels[framework_idx]}")
    
    # Add labels and title
    plt.xlabel('Communication Round Time (in sec)')
    plt.ylabel('Loss')
    plt.title(f"{title} ({n_clients} clients total)")
    plt.legend()
    plt.grid(True)

    plt.savefig(f'plots/test_loss_{test_type}_{n_clients}clnts.png')


def plot_comm_times(comm_times, test_type, labels, n_clients, title):
    os.makedirs('plots', exist_ok=True)  # creates 'plots/' if it doesn't exist
    # Plot the losses
    plt.figure(figsize=(10, 6))
    
    for framework_idx, framework_comm_times in enumerate(comm_times):
        plt.plot(framework_comm_times, label = f"{labels[framework_idx]}")
    
    # Add labels and title
    plt.xlabel('Communication Rounds')
    plt.ylabel('Communication Times')
    plt.title(f"{title} ({n_clients} clients total)")
    plt.legend()
    plt.grid(True)

    plt.savefig(f'plots/test_comm_times_{test_type}_{n_clients}clnts.png')


def test_cluster_framework_variations(test_type, args):
    n_clients = 50
    
    # get all of the client data and the testloader
    client_data, testloader = build_mnist(n_clients, args.iid_alpha, args.batch_size, seed=args.seed)
    
    if test_type == "equal_num_of_clients_per_cluster":

        # ------------------------
        # creating the parameter server model for the fully decentralized arch
        decentralized_arch_server_model = ConvNet()
        # creating the parameter server model for the decentralized arch
        accuracies_decentralized_comm, losses_decentralized_comm, times_decentralized_comm = fl_fullydecentralized_cluster_train(decentralized_arch_server_model, client_data, args.comm_rounds, args.lr, args.momentum, args.local_iters, args.straggler_max_delay, testloader)
        # ------------------------

        # ------------------------
        # creating the parameter server model for the cluster arch
        cluster_arch_server_model = ConvNet()
        # our framework
        accuracies_cluster_comm, losses_cluster_comm, times_cluster_comm = fl_semidecentralized_cluster_train(cluster_arch_server_model, client_data, args.comm_rounds, args.lr, args.momentum, args.local_iters, args.straggler_max_delay, testloader, "equal_num_of_clients_per_cluster", num_clusters=5)
        # ------------------------

        # ------------------------
        # creating the parameter server model for the central arch
        central_arch_server_model = ConvNet()
        # centralized fully sync SGD
        accuracies_central_comm, losses_central_comm, times_central_comm = fl_centralized_train(central_arch_server_model, client_data, args.comm_rounds, args.lr, args.momentum, args.local_iters, args.straggler_max_delay, testloader)
        # ------------------------

        plot_labels = ["Fully Decentralized Framework", "Our Semi-decentralized Framework", "Fully Sync SGD (centralized)"]
        accuracies = [accuracies_decentralized_comm, accuracies_cluster_comm, accuracies_central_comm]
        losses = [losses_decentralized_comm, losses_cluster_comm, losses_central_comm]
        times = [times_decentralized_comm, times_cluster_comm, times_central_comm]
        
        # plot everything
        plot_acc(accuracies, test_type, plot_labels, n_clients, 'Accuracy w/ eq num of clients per cluster')
        plot_loss(losses, test_type, plot_labels, n_clients, 'Loss w/ eq num of clients per cluster')
        plot_comm_times(times, test_type, plot_labels, n_clients, 'Communication Times w/ eq num of clients per cluster')

        
    # elif test_type == "diff_num_of_clusters":
    #     # TODO 
    #     pass
    # elif test_type == "diff_num_of_clients_per_cluster":
    #     # TODO 

    #     n_clients_values = [5, 10, 25, 50]

    #     for n_clients in n_clients_values:

    #         # get all of the client data and the testloader
    #         client_data, testloader = build_mnist(n_clients, args.iid_alpha, args.batch_size, seed=args.seed)
            
    #         # creating our parameter server model
    #         server_model = ConvNet()

    #         # train our framework
    #         # get the accuracies and timing
    #         accuracies_cluster_comm, times_cluster_comm = fl_cluster_train(server_model, client_data, args.comm_rounds, args.lr, args.momentum, args.local_iters, args.straggler_max_delay, testloader)


    #         # accuracies = fl_train(server_model, client_data, args.comm_rounds, args.lr, args.momentum, args.local_iters, testloader)
    #         # all_accuracies.append(accuracies)
        
    else:
        print("Unknown test type")



def test_cluster_comm_vs_gossip_comm(args):

    n_clients_values = [5, 10, 25, 50]

    for n_clients in n_clients_values:

        # get all of the client data and the testloader
        client_data, testloader = build_mnist(n_clients, args.iid_alpha, args.batch_size, seed=args.seed)
        
        # creating our parameter server model
        server_model = ConvNet()

        # train our framework
        # get the accuracies and timing
        accuracies_cluster_comm, times_cluster_comm = fl_semidecentralized_cluster_train(server_model, client_data, args.comm_rounds, args.lr, args.momentum, args.local_iters, args.straggler_max_delay, testloader, "equal_num_of_clients_per_cluster", num_clusters=5)

        # train centralized fedavg framework
        # get the accuracies and timing
        # accuracies_gossip_comm, times_gossip_comm = fl_gossip_train()


        # accuracies = fl_train(server_model, client_data, args.comm_rounds, args.lr, args.momentum, args.local_iters, testloader)
        # all_accuracies.append(accuracies)




def main(args): 
    print("Beginning Tests")
    # TODO: Will most likely have to move this.... or adapt it to capture the information from each test to properly plot it
    all_accuracies = [] # captures all of the accuracies
    all_times = [] # caputure the time it took to complete each iteration (this can be the avg time it took to complete the local iters, send data back to server, compute the aggregation, then send the new model back)

    # TESTING OUR FRAMWORK WITH VARIATIONS OF THE ARCHITECTURE
    # test equal amount of number of clients per cluster
    print("TEST 1: Testing when there are an equal amount of clients in a cluster vs the baseline fully sunc SGD")
    test_cluster_framework_variations("equal_num_of_clients_per_cluster", args)
    print("Done with TEST 1")
    
    # # test the number of clients in a given cluster (iid spread of clients in a cluster vs non iid)
    # test_cluster_framework_variations("diff_num_of_clients_per_cluster", args)
    
    # # test different number of clusters
    # test_cluster_framework_variations("diff_num_of_clusters", args)

    # # TESTING OUR FRAMWORK VS GOSSIP COMM
    # test_cluster_comm_vs_gossip_comm(args)


    




    # Get proper plots of the training

    # # Plot the accuracies
    # plt.figure(figsize=(10, 6))
    
    # # Loop over each accuracy list and corresponding n_clients value
    # for i, accuracies in enumerate(all_accuracies):
    #     plt.plot(accuracies, label=f'{n_clients_values[i]} clients')
    
    # # Add labels and title
    # plt.xlabel('Communication Rounds')
    # plt.ylabel('Accuracy')
    # plt.title('FedAvg w/ Different Numbers of Clients')
    # plt.legend()
    # plt.grid(True)
    
    # # Save the plot
    # plt.savefig('plots/question_1c.png')



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Project 18667 Cluster-Comm implementation', formatter_class=RawTextHelpFormatter)
    
    # System params 
    parser.add_argument('--seed', type=int, default=123,
        help='Torch random seed')
    
    # Dataset params
    parser.add_argument('--num-clients', type=int, default=10,
        help='Number of client devices to use in FL training')
    parser.add_argument('--iid-alpha', type=float, default=-1,
        help='Level of heterogeneity to introduce across client devices')
    parser.add_argument('--batch-size', type=int, default=32, 
        help='Batch size for local client training')
    
    # Server training params 
    parser.add_argument('--comm-rounds', type=int, default=30,
        help='Number of communication rounds')
    parser.add_argument('--clients-frac', type=int, default=1.0,
        help='Fraction of clients to use in each communication round')
    
    # Client training params 
    parser.add_argument('--lr', type=float, default=1e-3,
        help='Learning rate at client for local updates')
    parser.add_argument('--momentum', type=float, default=0.9,
        help='Momentum')
    parser.add_argument('--local-iters', type=int, default=10,
        help='Number of local iterations to use for training')
    parser.add_argument('--straggler-max-delay', type=int, default=5,
        help='The maximum time that a straggler can delay for')
    
    args = parser.parse_args()

    main(args)