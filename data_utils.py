import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
from scipy import stats
import random

def build_mnist(n_clients, alpha, batch_size, seed):
    """
    Builds the MNIST dataset for the various test scenarios we have.
    
    Args:
        n_clients (int): The number of clients for federated learning. If 1, centralized training is performed.
        alpha (float): The parameter for controlling the Dirichlet distribution used in partitioning the dataset 
                       among clients
        batch_size (int): The size of the batches to be loaded by the DataLoader.
        seed (int): torch random seed 
        
    Returns:
        Tuple[List[DataLoader], DataLoader]: Returns a tuple containing a list of DataLoaders for each client 
                                                 (training data) and a single DataLoader for testing data.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    Y = np.array(trainset.targets)

    clients = partition_dataset(trainset, Y, n_clients, alpha)
    clientloaders = [DataLoader(client, batch_size=batch_size, shuffle=True) for client in clients]
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    return clientloaders, testloader

# ------------ Helper functions for partition_dataset

def find_class_indexes(data_targets, shuffle=False):
    """
    Finds and groups the indexes from the original dataset according to class.
    Args:
        data_targets (np.array): The target labels of the dataset (helping with the grouping)
        shuffle (bool): Whether or not the indeces should be shuffled
    
    Returns:
        List[[ints]]: This is an i (where i can be 0 to max number of classes) by j (where j is the indices 
                      where the class label was at in the larger dataset) list. It contains the indices 
                      associated with each class.  
                      ex: arr[0] contains the indexes of the larger dataset where datapoints 
                      where the class label is 0.
    """
    num_classes = len(set(data_targets))
    class_indexes = [[] for i in range(num_classes)]
    for idx, class_num in enumerate(data_targets):
        class_indexes[class_num].append(idx)
    
    if shuffle:
        for i in range(len(class_indexes)):
            random.shuffle(class_indexes[i])
    
    return class_indexes


# # used for when we are in the Non IID case
# def split_data_by_proportions(class_specific_data, proportions):
#     """
#     Helps split the data in a non iid way according to proportions given by the Dirichlet distribution

#     Args:
#         class_specific_data (List[ints]): Contains the indexes of data specific to one class
#         proportions (np.array): Contains the proportions to help separate the data in a non 
#                                 iid way according to the Dirichlet distribution

#     Returns:
#         List[ints]: list full of the class indexes split according to the proportions given

#     """
#     num_of_data_points = len(class_specific_data)

#     class_specific_data = np.array(class_specific_data)
#     data_sizes = (proportions * num_of_data_points).astype(int)

#     # takes care of any rounding errors... making sure the total size remains equal to num_of_data_points
#     while data_sizes.sum() < num_of_data_points:
#         data_sizes[np.argmax(proportions)] += 1
#     while data_sizes.sum() > num_of_data_points:
#         data_sizes[np.argmax(data_sizes)] -= 1

#     # now split the data
#     splits = []
#     start = 0
#     for size in data_sizes:
#         splits.append(class_specific_data[start:start+size])
#         start += size

#     return splits
# ------------------------------------------------

def partition_dataset(dataset, Y, n_clients, alpha = -1):
    """
    Partitions a dataset into subsets for multiple clients, supporting both IID and non-IID cases.

    Args:
        dataset (torch.utils.data.Dataset): The original dataset to be partitioned.
        Y (np.array): The target labels of the dataset, used to group examples by class.
        n_clients (int): The number of clients.
        alpha (float): The parameter controlling the distribution of data across clients. 
                       If `alpha == -1`, the dataset is partitioned IID (Independent and Identically Distributed) which is the default.
                       If `alpha > 0`, the dataset is partitioned non-IID using a Dirichlet distribution
    
    Returns:
        List[torch.utils.data.Subset]: A list of `torch.utils.data.Subset` objects, where each subset represents the 
                                       data assigned to a particular client.
    """

    # print(f"The length of MNIST is {len(dataset)}")

    clients = []
    
    # section out separate arrays for each client
    data_idxs_for_each_client = [[] for i in range(n_clients)]

    # get the indexes for each class
    # want to ensure that the indexes are shuffled (have shuffled = true)
    class_indexes = find_class_indexes(Y, shuffle=True)

    # Only exploring the IID Case
    if alpha == -1:
        for class_idx_array in class_indexes:
            class_idx_array_split = np.array_split(np.array(class_idx_array), indices_or_sections=n_clients)
            for idx, single_class_partition_array in enumerate(class_idx_array_split):
                # The indexes for this class that we need
                needed_indexes = single_class_partition_array

                # after each iteration, a different partition of indexes from a class will be added to a different client
                # ex: if class 1 has 10 data points to be divided amongst 2 clients, this will add 5 class 1 data points to client 1 and 5 to client 2
                #     it will do so for all classes
                data_idxs_for_each_client[idx].extend(needed_indexes)
                
        

    # # NIID Case
    # else:

    #     for class_idx_array in class_indexes:
    #         alpha_vector = [alpha for i in range(n_clients)]
    #         class_data_ditr = np.random.dirichlet(alpha_vector)
    #         class_idx_array_split = split_data_by_proportions(class_idx_array, class_data_ditr)
    #         for idx, single_class_partition_array in enumerate(class_idx_array_split):
    #             # The indexes for this class that we need
    #             needed_indexes = single_class_partition_array

    #             # after each iteration, a different partition of indexes from a class will be added to a different client (but according to the proportions)
    #             data_idxs_for_each_client[idx].extend(needed_indexes)

    
    # add the data to the client array
    clients = [Subset(dataset, idx) for idx in data_idxs_for_each_client]

    return clients
