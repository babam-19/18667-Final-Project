import argparse
from argparse import RawTextHelpFormatter
import numpy as np
import matplotlib.pyplot as plt
import os

from models import ConvNet
from data_utils import build_mnist
from train_utils import fl_semidecentralized_cluster_train, fl_centralized_train, fl_fullydecentralized_cluster_train



def plot_acc(accuracies, test_type, labels, n_clients, title):
    os.makedirs(f'plots/{test_type}', exist_ok=True)  # creates 'plots/{test_type}' if it doesn't exist to separate depending on what test we are using
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

    plt.savefig(f'plots/{test_type}/test_accuracy_{test_type}_{n_clients}clnts.png')

def plot_loss(losses, test_type, labels, n_clients, title):
    os.makedirs(f'plots/{test_type}', exist_ok=True)  # creates 'plots/{test_type}' if it doesn't exist to separate depending on what test we are using
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


def test_cluster_framework_variations(test_type, args):
    
    if test_type == "T1_eval_against_baselines":

        # Test Description:
        # In this test, we will be evaluating our framework's performance against the baselines (being decentralized server architectire and
        # a central server gossip network architecture). We evaluate the accuracy, loss, and communication round time, and plot the 
        # results.

        n_clients = 50
    
        # get all of the client data and the testloader
        client_data, testloader = build_mnist(n_clients, args.iid_alpha, args.batch_size, seed=args.seed)

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
        accuracies_cluster_comm, losses_cluster_comm, times_cluster_comm = fl_semidecentralized_cluster_train(cluster_arch_server_model, client_data, args.comm_rounds, args.lr, args.momentum, args.local_iters, args.straggler_max_delay, testloader, "T1_eval_against_baselines", num_clusters=5)
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

    elif test_type == "T2_change_num_of_clusters":

        # Test Description:
        # In this test, we will be evaluating our framework's performance when you change the number of clusters used against the 
        # central server gossip network architecture. There are 50 clients total, but for our architecture, we use the following 
        # number of clusters: 5 clusters (10 clients per cluster), 10 clusters (5 clients per cluster), and 25 clusters (2 clients
        # per cluster). We evaluate the accuracy, loss, and communication round time, and plot the results.

        n_clients = 50
    
        # get all of the client data and the testloader
        client_data, testloader = build_mnist(n_clients, args.iid_alpha, args.batch_size, seed=args.seed)

        # ------------------------
        # creating the parameter server model for the cluster arch with 5 clusters (10 clients per cluster)
        cluster_arch_server_model_5 = ConvNet()
        # our framework
        accuracies_cluster_comm_5, losses_cluster_comm_5, times_cluster_comm_5 = fl_semidecentralized_cluster_train(cluster_arch_server_model_5, client_data, args.comm_rounds, args.lr, args.momentum, args.local_iters, args.straggler_max_delay, testloader, "T2_change_num_of_clusters", num_clusters=5)
        # ------------------------

        # ------------------------
        # creating the parameter server model for the cluster arch with 10 clusters (5 clients per cluster)
        cluster_arch_server_model_10 = ConvNet()
        # our framework
        accuracies_cluster_comm_10, losses_cluster_comm_10, times_cluster_comm_10 = fl_semidecentralized_cluster_train(cluster_arch_server_model_10, client_data, args.comm_rounds, args.lr, args.momentum, args.local_iters, args.straggler_max_delay, testloader, "T2_change_num_of_clusters", num_clusters=10)
        # ------------------------

        # ------------------------
        # creating the parameter server model for the cluster arch with 25 clusters (2 clients per cluster)
        cluster_arch_server_model_25 = ConvNet()
        # our framework
        accuracies_cluster_comm_25, losses_cluster_comm_25, times_cluster_comm_25 = fl_semidecentralized_cluster_train(cluster_arch_server_model_25, client_data, args.comm_rounds, args.lr, args.momentum, args.local_iters, args.straggler_max_delay, testloader, "T2_change_num_of_clusters", num_clusters=25)
        # ------------------------

        # ------------------------
        # creating the parameter server model for the central arch
        central_arch_server_model = ConvNet()
        # centralized fully sync SGD
        accuracies_central_comm, losses_central_comm, times_central_comm = fl_centralized_train(central_arch_server_model, client_data, args.comm_rounds, args.lr, args.momentum, args.local_iters, args.straggler_max_delay, testloader)
        # ------------------------

        plot_labels = ["Our Semi-decentralized Framework (clusters = 5)", "Our Semi-decentralized Framework (clusters = 10)",  "Our Semi-decentralized Framework (clusters = 25)", "Fully Sync SGD (centralized)"]
        accuracies = [accuracies_cluster_comm_5, accuracies_cluster_comm_10, accuracies_cluster_comm_25, accuracies_central_comm]
        losses = [losses_cluster_comm_5, losses_cluster_comm_10, losses_cluster_comm_25, losses_central_comm]
        times = [times_cluster_comm_5, times_cluster_comm_10, times_cluster_comm_25, times_central_comm]
        
        # plot everything
        plot_acc(accuracies, test_type, plot_labels, n_clients, 'Accuracy w/ adapting num of clusters')
        plot_loss(losses, test_type, plot_labels, n_clients, 'Loss w/ adapting num of clusters')
        plot_comm_times(times, test_type, plot_labels, n_clients, 'Communication Times w/ adapting num of clusters')

    elif test_type == "T3_scale_up_num_of_clients":

        # Test Description:
        # In this test, we will be evaluating our framework's performance when we scale up the number of clients against the baseline
        # central server gossip network architecture. We scale up the number of clients to 150. We evaluate the accuracy, loss, and 
        # communication round time, and plot the results.

        n_clients = 150
    
        # get all of the client data and the testloader
        client_data, testloader = build_mnist(n_clients, args.iid_alpha, args.batch_size, seed=args.seed)

        # ------------------------
        # creating the parameter server model for the cluster arch
        cluster_arch_server_model = ConvNet()
        # our framework
        accuracies_cluster_comm, losses_cluster_comm, times_cluster_comm = fl_semidecentralized_cluster_train(cluster_arch_server_model, client_data, args.comm_rounds, args.lr, args.momentum, args.local_iters, args.straggler_max_delay, testloader, "T3_scale_up_num_of_clients", num_clusters=10)
        # ------------------------

        # ------------------------
        # creating the parameter server model for the central arch
        central_arch_server_model = ConvNet()
        # centralized fully sync SGD
        accuracies_central_comm, losses_central_comm, times_central_comm = fl_centralized_train(central_arch_server_model, client_data, args.comm_rounds, args.lr, args.momentum, args.local_iters, args.straggler_max_delay, testloader)
        # ------------------------

        plot_labels = ["Our Semi-decentralized Framework (clusters = 10)", "Fully Sync SGD (centralized)"]
        accuracies = [accuracies_cluster_comm, accuracies_central_comm]
        losses = [losses_cluster_comm, losses_central_comm]
        times = [times_cluster_comm, times_central_comm]
        
        # plot everything
        plot_acc(accuracies, test_type, plot_labels, n_clients, 'Accuracy w/ scaled up num of clients (150)')
        plot_loss(losses, test_type, plot_labels, n_clients, 'Loss w/ scaled up num of clients (150)')
        plot_comm_times(times, test_type, plot_labels, n_clients, 'Communication Times w/ scaled up num of clients (150)')
        

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



def main(args): 
    print("Beginning Tests")

    # TESTING OUR FRAMWORK WITH VARIATIONS OF THE ARCHITECTURE

    # test our framework against the other baselines
    # print("TEST 1: Testing when there are an equal amount of clients in a cluster vs the baseline fully sync SGD and decentralized FL")
    # test_cluster_framework_variations("T1_eval_against_baselines", args)
    # print("Done with TEST 1")
    # print("---------------------------------")

    # test our framework with adapted num of clusters against one of the baselines
    print("TEST 2: Testing when the number of clusters are adapted (changing the num of clusters) vs the baseline fully sync SGD")
    test_cluster_framework_variations("T2_change_num_of_clusters", args)
    print("Done with TEST 2")
    print("---------------------------------")

    # # test our framework with scaled up num of clients against one of the baselines
    # print("TEST 3: Testing when we scale up the number of clients vs the baseline fully sync SGD")
    # test_cluster_framework_variations("T3_scale_up_num_of_clients", args)
    # print("Done with TEST 3")
    # print("---------------------------------")




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Project 18667 Cluster-Comm implementation', formatter_class=RawTextHelpFormatter)
    
    # System params 
    parser.add_argument('--seed', type=int, default=123,
        help='Torch random seed')
    
    # Dataset params
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
    parser.add_argument('--straggler-max-delay', type=int, default=0,
        help='The maximum time that a straggler can delay for')
    
    args = parser.parse_args()

    main(args)