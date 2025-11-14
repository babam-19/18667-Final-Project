import torch 
import numpy as np
from tqdm import tqdm
import copy 
# BA:
import torch.optim as optim
import random
from scipy.stats import truncexpon


def sample_straggler_delay(max_delay, mean_delay, size):
    """Helps to sample the straggler communication delay times while bounding the max time"""
    b = max_delay / mean_delay
    return truncexpon(b=b, scale=mean_delay).rvs(size)

def train_cluster(cluster_idx, clients_per_cluster, clients_in_clusters_ls, client_opts_in_clusters_ls, client_data_in_clusters_ls, local_iters, straggler_max_delay):
    num_clients_in_current_cluster = clients_per_cluster[cluster_idx]
    clients_in_current_cluster = clients_in_clusters_ls[cluster_idx]
    client_opts_in_current_cluster = client_opts_in_clusters_ls[cluster_idx]
    client_data_for_current_cluster = client_data_in_clusters_ls[cluster_idx]

    total_training_loss = []
    # TODO: wait on implementing the straggler time delay
    for model_idx in range(num_clients_in_current_cluster):
        # get the model specific information then train the model
        trainloader = client_data_for_current_cluster[model_idx]
        model = clients_in_current_cluster[model_idx]
        opt = client_opts_in_current_cluster[model_idx]
        training_loss = []

        # start training
        for i, (batch_input, batch_target) in enumerate(trainloader):
            # break if we reach the max number of iters
            if i == local_iters:
                break
            
            # get the prediction and the loss
            batch_pred = model(batch_input)
            loss = torch.nn.functional.cross_entropy(batch_pred, batch_target)
            training_loss.append(loss.detach().numpy()) # help for me to keep track of the training process
            
            # find the gradient and backprop
            opt.zero_grad()
            loss.backward()

            # update params
            opt.step()
        
        total_training_loss.append(np.mean(training_loss))

    return np.mean(total_training_loss)

def get_mixing_matrix(graph_type, num_clients):
    if graph_type == 'dense':
        # mixing matrix for dense communication graph (which is the only one we are considering in this case)
        mixing_matrix = np.ones((num_clients, num_clients)) / num_clients # TODO: change to a torch tensor for speed?
    return mixing_matrix

def exchange_cluster_to_cluster_info(cluster_idx, clients_per_cluster, clients_in_clusters_ls):
    num_clients_in_current_cluster = clients_per_cluster[cluster_idx]
    clients_in_current_cluster = clients_in_clusters_ls[cluster_idx]
    mixing_matrix = get_mixing_matrix("dense", num_clients_in_current_cluster)

    with torch.no_grad():
        # grabbing each param from each model
        client_params = []
        for model in clients_in_current_cluster:
            params = {name: param.clone().detach() for name, param in model.named_parameters()}
            client_params.append(params)
        
        # mixing eeach layer's weights and adding to mixed_params
        mixed_params = []
        for i in range(num_clients_in_current_cluster):
            mixed = {}
            for name in client_params[0].keys():
                
                # stacking all client parameters for a given layer
                stacked = torch.stack([client_params[j][name] for j in range(num_clients_in_current_cluster)])
                
                # weighted sum using mixing matrix row i
                mixed[name] = sum(mixing_matrix[i, j] * stacked[j] for j in range(num_clients_in_current_cluster))
           
            mixed_params.append(mixed)

        for i, model in enumerate(clients_in_current_cluster):
            for name, param in model.named_parameters():
                param.data.copy_(mixed_params[i][name])

def agg_models_at_ps(server_model, randomly_chosen_clients, clients_in_clusters_ls):

    """
    Aggregates the parameters of multiple client models by averaging them and updating the server model with the averaged parameters.
    After the aggregation, the new model is sent back to the randomly chosen client
    
    Args:
        server_model (torch.nn.Module): The server model that will be updated with the averaged parameters from the client models.
        randomly_chosen_client_idxs (List[int]): List containing the indexes of all the clients randomly chosen from a cluster
        clients_in_clusters_ls (List[torch.nn.Module]): List containing all of the clients in a given cluster
    
    Returns:
        None
    """
    # grabbing the clients from each cluster that will be used for aggregation
    clients_to_be_agg = []
    for cluster_idx, client_idx in enumerate(randomly_chosen_clients):
        clients_to_be_agg.append(clients_in_clusters_ls[cluster_idx][client_idx])

    # starting with 0 (when you see the zeros_like) for the mean of the weights in each layer 
    # planning on using Welford’s method for averaging
    all_model_params = [torch.zeros_like(parameter_in_layer) for parameter_in_layer in clients_to_be_agg[0].parameters()]
    
    # append all of the weights to the list, and find the running mean
    num_avged_client_params = 0 # how many clients we have used for the running avg calculation
    for model in clients_to_be_agg:
        # going through all of the layers to get the parameters (weights and the biases)
        num_avged_client_params += 1
        for layer_idx, (name, parameter) in enumerate(model.named_parameters()):
            all_model_params[layer_idx] += (parameter - all_model_params[layer_idx]) / num_avged_client_params

    with torch.no_grad():
        # now assign the new averages to the server model
        for layer_idx, (name, parameter) in enumerate(server_model.named_parameters()):
            parameter.copy_(all_model_params[layer_idx])

        # now give the new aggregated model to the randomly chosen client from each cluster
        for model in clients_to_be_agg:
            for layer_idx, (name, parameter) in enumerate(model.named_parameters()):
                parameter.copy_(all_model_params[layer_idx])

    

# def find_true_client_idx(cluster_to_cluster_client_idxs, clients_per_cluster):
#     """Will get the correct indexes for each client out of all the clients (not just from the clients in a goven cluster)"""
#     true_client_idxs = []
#     for cluster_idx, client_idx in enumerate(cluster_to_cluster_client_idxs):
#         if cluster_idx == 0:
#             true_client_idxs.append(client_idx)
#         else:
#             true_client_idxs.append(sum(clients_per_cluster[:cluster_idx]) + client_idx)

#     return true_client_idxs


def fl_cluster_train(server_model, client_data, comm_rounds, lr, momentum, local_iters, straggler_max_delay, testloader, main_test_type = "equal_num_of_clients_per_cluster", num_clusters = 5):
    accuracies = []
    times = []

    print("In function: fl_cluster_train")
    print(f"Communication Rounds = {comm_rounds}")
    print(f"Local iterations = {local_iters}\n\n")
    
    # initially making the copy of the server model for each client
    num_of_clients = len(client_data)
    client_list = [copy.deepcopy(server_model) for i in range(num_of_clients)]
    client_opts = [optim.SGD(a_client.parameters(), lr=lr, momentum=momentum) for a_client in client_list]
    
    # getting the proper number of clients in a cluster depending on the test type
    clients_in_clusters_ls = None
    client_opts_in_clusters_ls = None
    clients_per_cluster = None
    if main_test_type == "equal_num_of_clients_per_cluster":
        clients_in_clusters_ls = np.array_split(np.array(client_list), indices_or_sections=num_clusters)
        client_opts_in_clusters_ls = np.array_split(np.array(client_opts), indices_or_sections=num_clusters)
        client_data_in_clusters_ls = np.array_split(np.array(client_data), indices_or_sections=num_clusters)
        clients_per_cluster = [len(single_cluster) for single_cluster in clients_in_clusters_ls]
    elif main_test_type == "diff_num_of_clients_per_cluster":
        # TODO
        pass
    elif main_test_type == "diff_num_of_clusters":
        # TODO
        pass
    
    print(f"Num clients = {num_of_clients} with {num_clusters} clusters and about {np.mean(clients_per_cluster)} clients to a cluster")
    print(f"Num participating clients = {num_of_clients}")



    for current_round in range(comm_rounds):
        # will contain [chosen_client_idx_in_the_cluster]
        # ex: [5, 7, ...] means that from cluster 0, the client chosen for aggregation is client 5
        #                            from cluster 1, the client chosen for aggregation is client 7... etc
        randomly_chosen_clients = []

        # train each cluster and exchange the information amongst the clients in a cluster
        per_cluster_loss = []
        for cluster_idx in range(num_clusters):
            cluster_loss = train_cluster(cluster_idx, clients_per_cluster, clients_in_clusters_ls, client_opts_in_clusters_ls, client_data_in_clusters_ls, local_iters, straggler_max_delay)
            per_cluster_loss.append(cluster_loss)
            exchange_cluster_to_cluster_info(cluster_idx, clients_per_cluster, clients_in_clusters_ls)
            
             # pick a random client from a cluster to use for parameter server aggregation
            num_clients_in_cluster = clients_per_cluster[cluster_idx]
            randomly_chosen_clients.append(np.random.randint(0, num_clients_in_cluster))
        
        # aggregate at the PS and update the randomly chosen models
        agg_models_at_ps(server_model, randomly_chosen_clients, clients_in_clusters_ls)


        # test/evaluation
        validation_acc = []
        validation_loss = []
        for i, (test_batch_input, test_batch_target) in enumerate(testloader):
            # get the prediction from the server model
            test_batch_pred = server_model(test_batch_input)

            val_loss = torch.nn.functional.cross_entropy(test_batch_pred, test_batch_target)
            validation_loss.append(val_loss.detach().numpy()) # help for me to keep track of the validation loss
            
            # append the accuracy to the validation accuracy
            test_batch_pred_labels = torch.argmax(test_batch_pred.detach(), dim=1).numpy() # get the labels for the prediction
            eval_accuracy = np.sum((test_batch_pred_labels == test_batch_target.numpy())/test_batch_input.shape[0])
            validation_acc.append(eval_accuracy)
        
        total_val_loss = np.mean(validation_loss)
        total_train_loss = np.mean(per_cluster_loss)
        total_accuracy = np.mean(validation_acc)
        accuracies.append(total_accuracy)

        print(f"ROUND {current_round} - Total Training Loss: {total_train_loss:.5f} / Val Loss: {total_val_loss:5f} / Accuracy: {total_accuracy*100:5f}%")
        print(f"Training Loss Per Cluster: ")
        for idx, loss in enumerate(per_cluster_loss):
            print(f"Cluster {idx} Loss = {loss}")

    return accuracies, times