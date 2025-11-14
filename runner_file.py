import argparse
from argparse import RawTextHelpFormatter
import torch.nn as nn
import torch.optim as optim
import random 
import matplotlib.pyplot as plt
import sys

from models import ConvNet
from data_utils import build_mnist
from train_utils import fl_cluster_train



def test_cluster_framework_variations(test_type, args):
    n_clients = 50
    
    # get all of the client data and the testloader
    client_data, testloader = build_mnist(n_clients, args.iid_alpha, args.batch_size, seed=args.seed)
    
    # creating our parameter server model
    server_model = ConvNet()
    if test_type == "equal_num_of_clients_per_cluster":
        accuracies_cluster_comm, times_cluster_comm = fl_cluster_train(server_model, client_data, args.comm_rounds, args.lr, args.momentum, args.local_iters, args.straggler_max_delay, testloader, "equal_num_of_clients_per_cluster", num_clusters=5)
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
        accuracies_cluster_comm, times_cluster_comm = fl_cluster_train(server_model, client_data, args.comm_rounds, args.lr, args.momentum, args.local_iters, args.straggler_max_delay, testloader, "equal_num_of_clients_per_cluster", num_clusters=5)

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
    print("TEST 1: Testing when there are an equal amount of clients in a cluster")
    test_cluster_framework_variations("equal_num_of_clients_per_cluster", args)
    print("DOne with TEST 1")
    
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