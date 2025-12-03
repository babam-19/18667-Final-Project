import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
from scipy import stats
import random

def build_mnist(n_clients, alpha, batch_size, seed):
    """
    Builds the MNIST dataset for either centralized or federated learning scenarios.
    
    Args:
        n_clients (int): The number of clients for federated learning. If 1, centralized training is performed.
        alpha (float): The parameter for controlling the Dirichlet distribution used in partitioning the dataset 
                       among clients
        batch_size (int): The size of the batches to be loaded by the DataLoader.
        seed (int): torch random seed 
        
    Returns:
        If `n_clients == 1`:
            Tuple[DataLoader, DataLoader]: Returns a tuple containing the training and testing DataLoader 
                                           for centralized training.
        If `n_clients > 1`:
            Tuple[List[DataLoader], DataLoader]: Returns a tuple containing a list of DataLoaders for each client 
                                                 (training data) and a single DataLoader for testing data.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    N = len(trainset)
    Y = np.array(trainset.targets)
    n_classes = 10
    
    # Centralized training case 
    if n_clients == 1: 
        trainloader = DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True)
        testloader = DataLoader(dataset=testset, batch_size=batch_size, shuffle=False)
        return trainloader, testloader

    clients = partition_dataset(trainset, Y, n_classes, n_clients, alpha, seed)
    clientloaders = [DataLoader(client, batch_size=batch_size, shuffle=True) for client in clients]
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    return clientloaders, testloader
    
def partition_dataset(dataset, Y, n_classes, n_clients, alpha, seed):
    """
    Partitions a dataset into subsets for multiple clients, supporting both IID and non-IID cases.

    Args:
        dataset (torch.utils.data.Dataset): The original dataset to be partitioned.
        Y (np.array): The target labels of the dataset, used to group examples by class.
        n_classes (int): The number of unique classes in the dataset (e.g., 10 for MNIST).
        n_clients (int): The number of clients.
        alpha (float): The parameter controlling the distribution of data across clients. 
                       If `alpha == -1`, the dataset is partitioned IID (Independent and Identically Distributed).
                       If `alpha > 0`, the dataset is partitioned non-IID using a Dirichlet distribution
        seed (int): torch random seed.
    
    Returns:
        List[torch.utils.data.Subset]: A list of `torch.utils.data.Subset` objects, where each subset represents the 
                                       data assigned to a particular client.
    """
    clients = []

    # IID Case
    if alpha == -1:
        # randomly shuffle the dataset
        indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(seed))

        # find the size of each subset
        subset_size = len(dataset) // n_clients

        # create n_clients subsets
        for i in range(n_clients):
            start = i*subset_size
            end = start + subset_size
            clients.append(torch.utils.data.Subset(dataset, indices[start:end]))

    # NIID Case
    else:
        # find the indices for each class
        class_indices = [np.where(Y == i)[0] for i in range(n_classes)]

        # shuffle the indices for each class
        np.random.seed(seed)
        for i in class_indices:
            np.random.shuffle(i)
        
        # create a list of indices for each client
        client_indices = [[] for i in range(n_clients)]
        for i in class_indices:
            # find the proportions of each class for each client
            proportions = np.random.dirichlet(alpha=np.ones(n_clients) * alpha)
            
            # find number of samples of each class for each client
            num_samples = (len(i) * proportions).astype(int)

            # find the indices for each client
            start = 0
            for j in range(n_clients):
                client_samples = num_samples[j]
                end = start + client_samples
                client_indices[j] += i[start:end].tolist()
                start = end

        # create the subsets for each client
        for i in range(n_clients):
            # shuffle the indices for the client
            np.random.shuffle(client_indices[i])
            clients.append(torch.utils.data.Subset(dataset, client_indices[i]))

    return clients
