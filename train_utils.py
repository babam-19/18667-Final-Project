import torch 
import numpy as np
import copy 
# BA:
import torch.optim as optim
from scipy.stats import truncexpon
import time


def sample_straggler_delay(max_delay, num_clients, mean_delay=0.5):
    """Helps to sample the straggler communication delay times while bounding the max time"""
    b = max_delay / mean_delay
    return truncexpon(b=b, scale=mean_delay).rvs(num_clients)

def train_cluster(cluster_idx, clients_per_cluster, clients_in_clusters_ls, client_opts_in_clusters_ls,
                  client_data_in_clusters_ls, local_iters, selected_client_indices=None):
    """
    Trains all clients within a specified cluster for a fixed number of local iterations 
    and returns the average training loss across those clients.

    Args:
        cluster_idx (int): Index of the cluster to train.
        clients_per_cluster (list[int]): List containing the number of clients in each cluster.
        clients_in_clusters_ls (list[list[torch.nn.Module]]): Nested list of client models 
            grouped by cluster.
        client_opts_in_clusters_ls (list[list[torch.optim.Optimizer]]): Nested list of optimizers 
            corresponding to each client model in each cluster.
        client_data_in_clusters_ls (list[list[DataLoader]]): Nested list of dataloaders containing 
            training data for each client in each cluster.
        local_iters (int): Maximum number of local training iterations per client.

    Returns:
        float: The mean training loss across all clients in the specified cluster.
    """

    num_clients_in_current_cluster = clients_per_cluster[cluster_idx]
    clients_in_current_cluster = clients_in_clusters_ls[cluster_idx]
    client_opts_in_current_cluster = client_opts_in_clusters_ls[cluster_idx]
    client_data_for_current_cluster = client_data_in_clusters_ls[cluster_idx]

    if selected_client_indices is None:
        selected_client_indices = list(range(num_clients_in_current_cluster))

    total_training_loss = []

    for model_idx in selected_client_indices:
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
            training_loss.append(loss.detach().numpy())  # track loss

            # find the gradient and backprop
            opt.zero_grad()
            loss.backward()

            # update params
            opt.step()

        total_training_loss.append(np.mean(training_loss))

    return np.mean(total_training_loss)

def train_clients(client_ls, client_opt_ls, client_data_ls, local_iters, selected_client_indices=None):
    if selected_client_indices is None:
        selected_client_indices = list(range(len(client_ls)))

    total_training_loss = []
    for model_idx in selected_client_indices:
        # get the model specific information then train the model
        trainloader = client_data_ls[model_idx]
        model = client_ls[model_idx]
        opt = client_opt_ls[model_idx]
        training_loss = []

        # start training
        for i, (batch_input, batch_target) in enumerate(trainloader):
            # break if we reach the max number of iters
            if i == local_iters:
                break

            # get the prediction and the loss
            batch_pred = model(batch_input)
            loss = torch.nn.functional.cross_entropy(batch_pred, batch_target)
            training_loss.append(loss.detach().numpy())

            # find the gradient and backprop
            opt.zero_grad()
            loss.backward()

            # update params
            opt.step()

        total_training_loss.append(np.mean(training_loss))

    return np.mean(total_training_loss)

def eval_model(model_to_eval, test_dataloader):
    """
    Evaluates a given model using test data (taking the validation accuracy and the loss of the model)
    
    Args:
        model_to_eval (torch.nn.Module): The model that will be evaluated 
        test_dataloader (DataLoader): Test data used for the evaluation/validation
    
    Returns:
        (int, int): validation accuracy, validation loss
    """
     # test/evaluation
    validation_acc = []
    validation_loss = []
    for i, (test_batch_input, test_batch_target) in enumerate(test_dataloader):
        # get the prediction from the server model
        test_batch_pred = model_to_eval(test_batch_input)

        val_loss = torch.nn.functional.cross_entropy(test_batch_pred, test_batch_target)
        validation_loss.append(val_loss.detach().numpy()) # help for me to keep track of the validation loss
        
        # append the accuracy to the validation accuracy
        test_batch_pred_labels = torch.argmax(test_batch_pred.detach(), dim=1).numpy() # get the labels for the prediction
        eval_accuracy = np.sum((test_batch_pred_labels == test_batch_target.numpy())/test_batch_input.shape[0])
        validation_acc.append(eval_accuracy)

    return np.mean(validation_acc), np.mean(validation_loss)

def build_undirected_neighbors(graph_type, n, k=2, p=0.2, seed=0):
    """
    Returns neighbors list for an undirected graph on n nodes.
    graph_type: 'ring', 'star', 'random'
    """
    rng = np.random.default_rng(seed)
    neighbors = [set() for _ in range(n)]

    def add_edge(i, j):
        if i == j:
            return
        neighbors[i].add(j)
        neighbors[j].add(i)

    if graph_type == "ring":
        if k < 2:
            k = 2
        half = max(1, k // 2)
        for i in range(n):
            for d in range(1, half + 1):
                add_edge(i, (i + d) % n)
                add_edge(i, (i - d) % n)

    elif graph_type == "star":
        for i in range(1, n):
            add_edge(0, i)

    elif graph_type == "random":
        for i in range(n):
            for j in range(i + 1, n):
                if rng.random() < p:
                    add_edge(i, j)
        # ensure no isolated nodes 
        for i in range(n):
            if len(neighbors[i]) == 0 and n > 1:
                j = int(rng.integers(0, n - 1))
                if j >= i:
                    j += 1
                add_edge(i, j)
    else:
        raise ValueError(f"Unsupported graph_type: {graph_type}")

    return [sorted(list(s)) for s in neighbors]

def get_mixing_matrix(graph_type, num_clients, k=2, p=0.2, seed=0):
    if graph_type == "dense":
        return torch.ones((num_clients, num_clients)) / num_clients

    neighbors = build_undirected_neighbors(graph_type, num_clients, k=k, p=p, seed=seed)
    deg = [len(neighbors[i]) for i in range(num_clients)]

    W = torch.zeros((num_clients, num_clients), dtype=torch.float32)

    for i in range(num_clients):
        for j in neighbors[i]:
            W[i, j] = 1.0 / (1.0 + max(deg[i], deg[j]))

    for i in range(num_clients):
        W[i, i] = 1.0 - W[i].sum()

    W = torch.clamp(W, min=0.0)
    row_sums = W.sum(dim=1, keepdim=True)
    W = W / row_sums 

    return W

def exchange_client_to_client_info(clients, n_clients, graph_type="dense", k=2, p=0.2, seed=0):
    mixing_matrix = get_mixing_matrix(graph_type, n_clients, k=k, p=p, seed=seed)

    with torch.no_grad():
        # grabbing each param from each model
        client_params = []
        for model in clients:
            params = {name: param.clone().detach() for name, param in model.named_parameters()}
            client_params.append(params)
        
        # mixing eeach layer's weights and adding to mixed_params
        mixed_params = []
        for i in range(n_clients):
            mixed = {}
            nz = torch.nonzero(mixing_matrix[i]).flatten().tolist()
            for name in client_params[0].keys():
                mixed[name] = sum(mixing_matrix[i, j] * client_params[j][name] for j in nz)
            mixed_params.append(mixed)

        for i, model in enumerate(clients):
            for name, param in model.named_parameters():
                param.data.copy_(mixed_params[i][name])

def agg_models_at_ps_for_cluster_arch(server_model, randomly_chosen_clients, clients_in_clusters_ls):
    """
    Aggregates the parameters of multiple client models by averaging them and updating the server model
    with the averaged parameters. After aggregation, the new model is sent back to the randomly chosen
    clients (one per cluster).

    Args:
        server_model (torch.nn.Module): The server model to update.
        randomly_chosen_clients (List[int]): For each cluster, the index of the chosen client within that cluster.
        clients_in_clusters_ls (List[List[torch.nn.Module]]): Client models grouped by cluster.

    Returns:
        None
    """

    # get the representative model from each cluster
    clients_to_be_agg = []
    for cluster_idx, client_idx in enumerate(randomly_chosen_clients):
        clients_to_be_agg.append(clients_in_clusters_ls[cluster_idx][client_idx])

    # running average of parameters (Welford-style mean)
    all_model_params = [torch.zeros_like(p) for p in clients_to_be_agg[0].parameters()]
    num_avged_client_params = 0

    for model in clients_to_be_agg:
        num_avged_client_params += 1
        for layer_idx, p in enumerate(model.parameters()):
            all_model_params[layer_idx] += (p - all_model_params[layer_idx]) / num_avged_client_params

    with torch.no_grad():
        # update server model
        for layer_idx, p in enumerate(server_model.parameters()):
            p.copy_(all_model_params[layer_idx])

        # push updated params back to the chosen reps
        for model in clients_to_be_agg:
            for layer_idx, p in enumerate(model.parameters()):
                p.copy_(all_model_params[layer_idx])

def agg_models_at_ps_for_central_arch(server_model, clients, selected_client_indices=None):
    """
    Aggregates the parameters of multiple client models by averaging them and updating the server model with the averaged parameters.
    After the aggregation, the new model is sent back to all of the clients
    
    Args:
        server_model (torch.nn.Module): The server model that will be updated with the averaged parameters from the client models.
        clients (List[torch.nn.Module]): List containing all of the clients 
    
    Returns:
        None
    """
    if selected_client_indices is None:
        selected_client_indices = list(range(len(clients)))

    clients_to_be_agg = [clients[i] for i in selected_client_indices]

    # Welford-style mean over selected clients
    all_model_params = [torch.zeros_like(p) for p in clients_to_be_agg[0].parameters()]
    num_avged_client_params = 0

    for model in clients_to_be_agg:
        num_avged_client_params += 1
        for layer_idx, p in enumerate(model.parameters()):
            all_model_params[layer_idx] += (p - all_model_params[layer_idx]) / num_avged_client_params

    with torch.no_grad():
        # update server model
        for layer_idx, p in enumerate(server_model.parameters()):
            p.copy_(all_model_params[layer_idx])

        # broadcast to all clients (including non-participants)
        for model in clients:
            for layer_idx, p in enumerate(model.parameters()):
                p.copy_(all_model_params[layer_idx])

def setup_client_cluster_arch(main_test_type, num_clusters, ls_of_clients, ls_of_client_opts, ls_of_client_data,
                              cluster_sizes=None, shuffle_clients=False, seed=0):
    base_test_type = main_test_type.split("__")[0]
    
    clients_in_clusters_ls = None
    client_opts_in_clusters_ls = None
    clients_per_cluster = None
    client_data_in_clusters_ls = None

    n = len(ls_of_clients)
    if shuffle_clients:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(n)
        ls_of_clients = [ls_of_clients[i] for i in perm]
        ls_of_client_opts = [ls_of_client_opts[i] for i in perm]
        ls_of_client_data = [ls_of_client_data[i] for i in perm]

    # tests in the main runner file that require an equal number of clients per cluster
    eq_clients_per_cluster = ["T1_eval_against_baselines", "T2_change_num_of_clusters", 
                              "T3_scale_up_num_of_clients", "T4_graph_connectivity",
                              "T5_global_sync_frequency", "diff_num_of_clusters",
                              "T8_partial_participation"]
    
    if base_test_type in eq_clients_per_cluster:
        clients_in_clusters_ls = np.array_split(np.array(ls_of_clients), indices_or_sections=num_clusters)
        client_opts_in_clusters_ls = np.array_split(np.array(ls_of_client_opts), indices_or_sections=num_clusters)
        client_data_in_clusters_ls = np.array_split(np.array(ls_of_client_data), indices_or_sections=num_clusters)
        clients_per_cluster = [len(single_cluster) for single_cluster in clients_in_clusters_ls]
    elif base_test_type == "diff_num_of_clients_per_cluster":
        # cluster_sizes must be provided: e.g. [5, 10, 7, 8, ...]
        if cluster_sizes is None:
            raise ValueError("diff_num_of_clients_per_cluster requires cluster_sizes.")

        cluster_sizes = list(cluster_sizes)
        num_clusters = len(cluster_sizes)
        total = int(np.sum(cluster_sizes))
        if total < n:
            cluster_sizes[-1] += (n - total)
        elif total > n:
            raise ValueError(f"cluster_sizes sum to {total} but there are only {n} clients.")

        clients_in_clusters_ls = []
        client_opts_in_clusters_ls = []
        client_data_in_clusters_ls = []

        start = 0
        for curr_size in cluster_sizes:
            end = start + curr_size
            clients_in_clusters_ls.append(np.array(ls_of_clients[start:end], dtype=object))
            client_opts_in_clusters_ls.append(np.array(ls_of_client_opts[start:end], dtype=object))
            client_data_in_clusters_ls.append(np.array(ls_of_client_data[start:end], dtype=object))
            start = end

        clients_per_cluster = cluster_sizes

    else:
        raise ValueError(f"Unknown main_test_type: {main_test_type}")

    return clients_in_clusters_ls, client_opts_in_clusters_ls, clients_per_cluster,  client_data_in_clusters_ls

def fl_semidecentralized_cluster_train(server_model, client_data, comm_rounds,
                                       lr, momentum, local_iters, straggler_max_delay,
                                       testloader, main_test_type="eval_against_baselines",
                                       num_clusters=5, intra_graph_type="dense", intra_graph_k=2,
                                       intra_graph_p=0.2, intra_graph_seed=0, gossip_rounds=1,
                                       global_sync_E=1, cluster_sizes=None, cluster_schedule=None,
                                       reshuffle_clusters_each_round=False, clients_frac=1.0):
    accuracies = []
    losses = []
    times = []

    print("In function: fl_semidecentralized_cluster_train")
    print(f"Communication Rounds = {comm_rounds}")
    print(f"Local iterations = {local_iters}\n\n")

    # initially making the copy of the server model for each client
    num_of_clients = len(client_data)
    client_list = [copy.deepcopy(server_model) for i in range(num_of_clients)]
    client_opts = [optim.SGD(a_client.parameters(), lr=lr, momentum=momentum) for a_client in client_list]

    # rng for participant sampling (reproducible)
    participation_rng = np.random.default_rng(intra_graph_seed)

    # getting the proper number of clients in a cluster depending on the test type
    base_test_type = main_test_type.split("__")[0]
    if cluster_schedule is None:
        clients_in_clusters_ls, client_opts_in_clusters_ls, clients_per_cluster, client_data_in_clusters_ls = setup_client_cluster_arch(
            main_test_type, num_clusters, client_list, client_opts, client_data,
            cluster_sizes=cluster_sizes,
            shuffle_clients=reshuffle_clusters_each_round,
            seed=intra_graph_seed
        )
        num_clusters_effective = len(clients_per_cluster)

        print(f"Num clients = {num_of_clients} with {num_clusters_effective} clusters and about {np.mean(clients_per_cluster)} clients to a cluster")
    else:
        print(f"Num clients = {num_of_clients} with dynamic clusters per round (schedule length = {len(cluster_schedule)})")

    # partial participation info
    clients_frac = float(clients_frac)
    if clients_frac <= 0.0 or clients_frac > 1.0:
        raise ValueError(f"clients_frac must be in (0, 1], got {clients_frac}")

    print(f"clients_frac = {clients_frac}")

    if straggler_max_delay > 0:
        print(f"Incorporating straggler delay (max straggler delay = {straggler_max_delay})")

    for current_round in range(comm_rounds):
        # dynamic re-clustering per round
        if cluster_schedule is not None:
            num_clusters_round = int(cluster_schedule[current_round])
            clients_in_clusters_ls, client_opts_in_clusters_ls, clients_per_cluster, client_data_in_clusters_ls = setup_client_cluster_arch(
                "diff_num_of_clusters",
                num_clusters_round,
                client_list,
                client_opts,
                client_data,
                shuffle_clients=reshuffle_clusters_each_round,
                seed=intra_graph_seed + current_round
            )
            num_clusters_effective = len(clients_per_cluster)
        else:
            num_clusters_effective = len(clients_per_cluster)

        # if we are using/enforcing straggler delay then get the times for each cluster
        if straggler_max_delay > 0:
            time_delay_per_cluster = []
            for cluster_sizes in clients_per_cluster:
                time_delay_per_cluster.append(np.sum(sample_straggler_delay(straggler_max_delay, cluster_sizes)))
            total_time_delay = np.sum(time_delay_per_cluster)

        # our starting time
        round_start_time = time.time()

        randomly_chosen_clients = []

        # train each cluster and exchange the information amongst the clients in a cluster
        per_cluster_loss = []
        for cluster_idx in range(num_clusters_effective):
            num_clients_in_current_cluster = clients_per_cluster[cluster_idx]
            clients_in_current_cluster = clients_in_clusters_ls[cluster_idx]

            # partial participation: pick subset in this cluster
            if clients_frac >= 1.0:
                participating_idx = list(range(num_clients_in_current_cluster))
            else:
                n_part = int(np.floor(clients_frac * num_clients_in_current_cluster))
                n_part = max(1, n_part)
                participating_idx = participation_rng.choice(
                    num_clients_in_current_cluster, size=n_part, replace=False
                ).tolist()

            # train only participating clients
            cluster_loss = train_cluster(
                cluster_idx, clients_per_cluster, clients_in_clusters_ls,
                client_opts_in_clusters_ls, client_data_in_clusters_ls, local_iters,
                selected_client_indices=participating_idx
            )
            per_cluster_loss.append(cluster_loss)

            # gossip only among participating clients (non-participants stay untouched this round)
            participating_models = [clients_in_current_cluster[i] for i in participating_idx]
            for _ in range(gossip_rounds):
                exchange_client_to_client_info(
                    participating_models, len(participating_models),
                    graph_type=intra_graph_type, k=intra_graph_k, p=intra_graph_p,
                    seed=intra_graph_seed + current_round + cluster_idx
                )

            # pick a random participating client from a cluster to use for parameter server aggregation
            chosen = int(participation_rng.choice(participating_idx, size=1)[0])
            randomly_chosen_clients.append(chosen)

        # aggregate at the PS only every global_sync_E rounds
        do_global_sync = (global_sync_E <= 1) or ((current_round + 1) % global_sync_E == 0)

        # build the representative models (one per cluster)
        reps = []
        for cluster_idx, client_idx in enumerate(randomly_chosen_clients):
            reps.append(clients_in_clusters_ls[cluster_idx][client_idx])

        # compute the averaged "global" params from representatives (Welford-style mean)
        avg_params = [torch.zeros_like(p) for p in reps[0].parameters()]
        num_avged = 0
        for m in reps:
            num_avged += 1
            for layer_idx, p in enumerate(m.parameters()):
                avg_params[layer_idx] += (p - avg_params[layer_idx]) / num_avged

        if do_global_sync:
            # real global sync: update server + push to reps
            with torch.no_grad():
                for layer_idx, p in enumerate(server_model.parameters()):
                    p.copy_(avg_params[layer_idx])
                for m in reps:
                    for layer_idx, p in enumerate(m.parameters()):
                        p.copy_(avg_params[layer_idx])
        else:
            # no global sync this round: evaluate the "virtual" global model without changing training state
            backup_server_state = copy.deepcopy(server_model.state_dict())
            with torch.no_grad():
                for layer_idx, p in enumerate(server_model.parameters()):
                    p.copy_(avg_params[layer_idx])

        # our ending time (changes depending on whether we are enforcing straggler delay)
        if straggler_max_delay > 0:
            round_end_time = time.time() + total_time_delay
        else:
            round_end_time = time.time()

        # test/evaluation
        total_accuracy, total_val_loss = eval_model(server_model, testloader)

        # restore server model
        if not do_global_sync:
            server_model.load_state_dict(backup_server_state)

        total_train_loss = np.mean(per_cluster_loss)

        accuracies.append(total_accuracy)
        losses.append(total_train_loss)
        times.append(round_end_time - round_start_time)

        print(f"ROUND {current_round} - Total Training Loss: {total_train_loss:.5f} / Val Loss: {total_val_loss:5f} / Accuracy: {total_accuracy*100:5f}%")
        print(f"Training Loss Per Cluster: ")
        for idx, loss in enumerate(per_cluster_loss):
            print(f"Cluster {idx} Loss = {loss}")

    return accuracies, losses, times

def fl_fullydecentralized_cluster_train(starting_model, client_data, comm_rounds, lr, momentum,
                                        local_iters, straggler_max_delay, testloader, clients_frac=1.0):
    accuracies = []
    losses = []
    times = []

    print("In function: fl_fullydecentralized_cluster_train")
    print(f"Communication Rounds = {comm_rounds}")
    print(f"Local iterations = {local_iters}\n\n")

    num_of_clients = len(client_data)
    client_list = [copy.deepcopy(starting_model) for i in range(num_of_clients)]
    client_opts = [optim.SGD(a_client.parameters(), lr=lr, momentum=momentum) for a_client in client_list]

    clients_frac = float(clients_frac)
    if clients_frac <= 0.0 or clients_frac > 1.0:
        raise ValueError(f"clients_frac must be in (0, 1], got {clients_frac}")

    print(f"Num clients = {num_of_clients}")
    print(f"clients_frac = {clients_frac}")

    rng = np.random.default_rng(0)

    if straggler_max_delay > 0:
        print(f"Incorporating straggler delay (max straggler delay = {straggler_max_delay})")

    # for evaluation: use a virtual averaged model (so metrics don’t depend on whether client 0 participated)
    virtual_model = copy.deepcopy(starting_model)

    for current_round in range(comm_rounds):
        
        # if we are using/enforcing straggler delay then get the times for each client
        if straggler_max_delay > 0:
            # get the total time delay for all of the clients
            total_time_delay = np.sum(sample_straggler_delay(straggler_max_delay, num_of_clients))

        round_start_time = time.time()

        # sample participating clients
        if clients_frac >= 1.0:
            participating = list(range(num_of_clients))
        else:
            n_part = int(np.floor(clients_frac * num_of_clients))
            n_part = max(1, n_part)
            participating = rng.choice(num_of_clients, size=n_part, replace=False).tolist()

        # train only participating clients
        all_client_loss = train_clients(client_list, client_opts, client_data, local_iters,
                                        selected_client_indices=participating)

        # gossip only among participating clients
        participating_models = [client_list[i] for i in participating]
        exchange_client_to_client_info(participating_models, len(participating_models))

        # our ending time (changes depending on whether we are enforcing straggler delay)
        if straggler_max_delay > 0:
            # if we enforce straggler delay, then add the additional time to reflect that
            round_end_time = time.time() + total_time_delay
        else:
            # otherwise use the time that was recorded as is
            round_end_time = time.time()

        # evaluate a virtual global model = average of all clients
        with torch.no_grad():
            avg_params = [torch.zeros_like(p) for p in client_list[0].parameters()]
            num_avged = 0
            for m in client_list:
                num_avged += 1
                for layer_idx, p in enumerate(m.parameters()):
                    avg_params[layer_idx] += (p - avg_params[layer_idx]) / num_avged
            for layer_idx, p in enumerate(virtual_model.parameters()):
                p.copy_(avg_params[layer_idx])

        total_accuracy, total_val_loss = eval_model(virtual_model, testloader)
        total_train_loss = all_client_loss

        accuracies.append(total_accuracy)
        losses.append(total_train_loss)
        times.append(round_end_time - round_start_time)

        print(f"ROUND {current_round} - Total Training Loss: {total_train_loss:.5f} / Val Loss: {total_val_loss:5f} / Accuracy: {total_accuracy*100:5f}%")

    return accuracies, losses, times

def fl_centralized_train(server_model, client_data, comm_rounds, lr, momentum, local_iters,
                         straggler_max_delay, testloader, clients_frac=1.0):
    accuracies = []
    losses = []
    times = []

    print("In function: fl_centralized_train")
    print(f"Communication Rounds = {comm_rounds}")
    print(f"Local iterations = {local_iters}\n\n")

    # initially making the copy of the server model for each client
    num_of_clients = len(client_data)
    client_list = [copy.deepcopy(server_model) for i in range(num_of_clients)]
    client_opts = [optim.SGD(a_client.parameters(), lr=lr, momentum=momentum) for a_client in client_list]

    clients_frac = float(clients_frac)
    if clients_frac <= 0.0 or clients_frac > 1.0:
        raise ValueError(f"clients_frac must be in (0, 1], got {clients_frac}")

    print(f"Num clients = {num_of_clients} ")
    print(f"clients_frac = {clients_frac}")

    rng = np.random.default_rng(0)

    if straggler_max_delay > 0:
        print(f"Incorporating straggler delay (max straggler delay = {straggler_max_delay})")


    for current_round in range(comm_rounds):
        
        # if we are using/enforcing straggler delay then get the times for each client
        if straggler_max_delay > 0:
            # get the total time delay for all of the clients
            total_time_delay = np.sum(sample_straggler_delay(straggler_max_delay, num_of_clients))

        # our starting time
        round_start_time = time.time()

        # sample participating clients (global)
        if clients_frac >= 1.0:
            participating = list(range(num_of_clients))
        else:
            n_part = int(np.floor(clients_frac * num_of_clients))
            n_part = max(1, n_part)
            participating = rng.choice(num_of_clients, size=n_part, replace=False).tolist()

        # train only participating clients
        all_client_loss = train_clients(client_list, client_opts, client_data, local_iters,
                                        selected_client_indices=participating)

        # average ONLY participating clients, then broadcast to all
        agg_models_at_ps_for_central_arch(server_model, client_list, selected_client_indices=participating)

        if straggler_max_delay > 0:
            round_end_time = time.time() + total_time_delay
        else:
            round_end_time = time.time()

        total_accuracy, total_val_loss = eval_model(server_model, testloader)
        total_train_loss = all_client_loss

        accuracies.append(total_accuracy)
        losses.append(total_train_loss)
        times.append(round_end_time - round_start_time)

        print(f"ROUND {current_round} - Total Training Loss: {total_train_loss:.5f} / Val Loss: {total_val_loss:5f} / Accuracy: {total_accuracy*100:5f}%")

    return accuracies, losses, times