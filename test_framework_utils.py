from models import ConvNet
from data_utils import build_mnist
from train_utils import fl_semidecentralized_cluster_train, fl_centralized_train, fl_fullydecentralized_cluster_train
from plot_utils import plot_acc, plot_loss, plot_comm_times

def test_cluster_framework_variations(test_type, args):
    
    if test_type.startswith("T1_eval_against_baselines"):
        # Test Description:
        # In this test, we will be evaluating our framework's performance against the baselines (being decentralized server architectire and
        # a central server gossip network architecture). We evaluate the accuracy, loss, and communication round time, and plot the 
        # results.

        n_clients = getattr(args, "n_clients", 50)
        num_clusters = getattr(args, "num_clusters", 5)
    
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
        accuracies_cluster_comm, losses_cluster_comm, times_cluster_comm = fl_semidecentralized_cluster_train(cluster_arch_server_model, client_data, args.comm_rounds, args.lr, args.momentum, args.local_iters, args.straggler_max_delay, testloader, test_type, num_clusters=num_clusters)
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

    elif test_type.startswith("T2_change_num_of_clusters"):

        # Test Description:
        # In this test, we will be evaluating our framework's performance when you change the number of clusters used against the 
        # central server gossip network architecture. There are 50 clients total, but for our architecture, we use the following 
        # number of clusters: 5 clusters (10 clients per cluster), 10 clusters (5 clients per cluster), and 25 clusters (2 clients
        # per cluster). We evaluate the accuracy, loss, and communication round time, and plot the results.

        n_clients = getattr(args, "n_clients", 50)
        cluster_counts = getattr(args, "cluster_counts", [5, 10, 25])
        accuracies, losses, times = [], [], []
    
        # get all of the client data and the testloader
        client_data, testloader = build_mnist(n_clients, args.iid_alpha, args.batch_size, seed=args.seed)

        # ------------------------
        
        for cluster_count in cluster_counts:
            # creating the parameter server model for the cluster arch with 5 clusters (10 clients per cluster)
            cluster_arch_server_model = ConvNet()
            # our framework
            accuracies_cluster_comm, losses_cluster_comm, times_cluster_comm = fl_semidecentralized_cluster_train(cluster_arch_server_model, client_data, args.comm_rounds, args.lr, args.momentum, args.local_iters, args.straggler_max_delay, testloader, test_type, num_clusters=cluster_count)
            accuracies.append(accuracies_cluster_comm)
            losses.append(losses_cluster_comm)
            times.append(times_cluster_comm)
        # ------------------------

        # ------------------------
        # creating the parameter server model for the central arch
        central_arch_server_model = ConvNet()
        # centralized fully sync SGD
        accuracies_central_comm, losses_central_comm, times_central_comm = fl_centralized_train(central_arch_server_model, client_data, args.comm_rounds, args.lr, args.momentum, args.local_iters, args.straggler_max_delay, testloader)
        # ------------------------

        plot_labels = [f"Our Semi-decentralized Framework (clusters = {c})" for c in cluster_counts]
        plot_labels.append("Fully Sync SGD (centralized)")
        accuracies.append(accuracies_central_comm)
        losses.append(losses_central_comm)
        times.append(times_central_comm)
        
        # plot everything
        plot_acc(accuracies, test_type, plot_labels, n_clients, 'Accuracy w/ adapting num of clusters')
        plot_loss(losses, test_type, plot_labels, n_clients, 'Loss w/ adapting num of clusters')
        plot_comm_times(times, test_type, plot_labels, n_clients, 'Communication Times w/ adapting num of clusters')

    elif test_type.startswith("T3_scale_up_num_of_clients"):
        # Test Description:
        # In this test, we will be evaluating our framework's performance when we scale up the number of clients against the baseline
        # central server gossip network architecture. We scale up the number of clients to 150. We evaluate the accuracy, loss, and 
        # communication round time, and plot the results.

        n_clients = getattr(args, "n_clients", 150)
        num_clusters = getattr(args, "num_clusters", 10)
    
        # get all of the client data and the testloader
        client_data, testloader = build_mnist(n_clients, args.iid_alpha, args.batch_size, seed=args.seed)

        # ------------------------
        # creating the parameter server model for the cluster arch
        cluster_arch_server_model = ConvNet()
        # our framework
        accuracies_cluster_comm, losses_cluster_comm, times_cluster_comm = fl_semidecentralized_cluster_train(cluster_arch_server_model, client_data, args.comm_rounds, args.lr, args.momentum, args.local_iters, args.straggler_max_delay, testloader, test_type, num_clusters=num_clusters)
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

    elif test_type.startswith("T4_graph_connectivity"):

        n_clients = getattr(args, "n_clients", 50)
        num_clusters = getattr(args, "num_clusters", 5)
        configs = getattr(args, "connectivity_configs", [])

        client_data, testloader = build_mnist(n_clients, args.iid_alpha, args.batch_size, seed=args.seed)

        accuracies, losses, times = [], [], []
        plot_labels = []

        for cfg in configs:
            label = cfg.get("label", cfg.get("graph_type", "unknown"))
            graph_type = cfg.get("graph_type", "dense")
            k = cfg.get("k", 2)
            p = cfg.get("p", 0.2)
            gossip_rounds = cfg.get("gossip_rounds", 1)

            server_model = ConvNet()
            acc, loss, t = fl_semidecentralized_cluster_train(
                server_model, client_data, args.comm_rounds,
                args.lr, args.momentum, args.local_iters, args.straggler_max_delay,
                testloader, main_test_type=test_type, num_clusters=num_clusters,
                intra_graph_type=graph_type, intra_graph_k=k, intra_graph_p=p,
                gossip_rounds=gossip_rounds, intra_graph_seed=args.seed
            )

            accuracies.append(acc)
            losses.append(loss)
            times.append(t)
            plot_labels.append(label)

        plot_acc(accuracies, test_type, plot_labels, n_clients, "Accuracy vs connectivity")
        plot_loss(losses, test_type, plot_labels, n_clients, "Loss vs connectivity")
        plot_comm_times(times, test_type, plot_labels, n_clients, "Comm time vs connectivity")

    elif test_type.startswith("T5_global_sync_frequency"):
        # Test Description:
        # Global synchronization frequency E (sync every E rounds)
        # plot accuracies for each E on the same graph.

        n_clients = getattr(args, "n_clients", 50)
        num_clusters = getattr(args, "num_clusters", 5)

        # List of E values
        E_values = getattr(args, "E_values", [1, 2, 5, 10])

        # Keep intra-cluster connectivity fixed 
        intra_graph_type = getattr(args, "intra_graph_type", "ring")
        intra_graph_k = getattr(args, "intra_graph_k", 2)
        intra_graph_p = getattr(args, "intra_graph_p", 0.1)
        gossip_rounds = getattr(args, "gossip_rounds", 1)

        client_data, testloader = build_mnist(n_clients, args.iid_alpha, args.batch_size, seed=args.seed)

        accuracies, losses, times = [], [], []
        plot_labels = []

        for E in E_values:
            server_model = ConvNet()

            acc, loss, t = fl_semidecentralized_cluster_train(
                server_model, client_data, args.comm_rounds, args.lr, args.momentum,
                args.local_iters, args.straggler_max_delay, testloader, main_test_type=test_type, 
                num_clusters=num_clusters, intra_graph_type=intra_graph_type, 
                intra_graph_k=intra_graph_k,  intra_graph_p=intra_graph_p, 
                intra_graph_seed=args.seed, gossip_rounds=gossip_rounds, global_sync_E=E
            )

            accuracies.append(acc)
            losses.append(loss)
            times.append(t)
            plot_labels.append(f"E={E}")

        plot_acc(accuracies, test_type, plot_labels, n_clients, "Accuracy vs global sync frequency (E)")
        plot_loss(losses, test_type, plot_labels, n_clients, "Loss vs global sync frequency (E)")
        plot_comm_times(times, test_type, plot_labels, n_clients, "Comm time vs global sync frequency (E)")

    elif test_type.startswith("T6_diff_num_of_clients_per_cluster"):
        # Test Description:
        # Fixed number of total clients, but uneven cluster sizes
        n_clients = getattr(args, "n_clients", 50)
        num_clusters = getattr(args, "num_clusters", 5)
        cluster_sizes = getattr(args, "cluster_sizes", None)

        client_data, testloader = build_mnist(n_clients, args.iid_alpha, args.batch_size, seed=args.seed)

        server_model = ConvNet()
        acc, loss, t = fl_semidecentralized_cluster_train(
            server_model, client_data, args.comm_rounds, args.lr, args.momentum, args.local_iters,
            args.straggler_max_delay, testloader, main_test_type="diff_num_of_clients_per_cluster", 
            num_clusters=num_clusters, cluster_sizes=cluster_sizes, 
            intra_graph_type=getattr(args, "intra_graph_type", "dense"), intra_graph_k=getattr(args, "intra_graph_k", 2), 
            intra_graph_p=getattr(args, "intra_graph_p", 0.2), intra_graph_seed=args.seed, 
            gossip_rounds=getattr(args, "gossip_rounds", 1), global_sync_E=getattr(args, "global_sync_E", 1),
        )

        plot_acc([acc], test_type, ["Uneven clusters"], n_clients, "Accuracy (uneven cluster sizes)")
        plot_loss([loss], test_type, ["Uneven clusters"], n_clients, "Loss (uneven cluster sizes)")
        plot_comm_times([t], test_type, ["Uneven clusters"], n_clients, "Comm time (uneven cluster sizes)")

    elif test_type.startswith("T7_diff_num_of_clusters_over_rounds"):
        # Test Description:
        # Change number of clusters over rounds using a schedule
        n_clients = getattr(args, "n_clients", 50)
        cluster_schedule = getattr(args, "cluster_schedule", None)
        if cluster_schedule is None:
            raise ValueError("T7 requires args.cluster_schedule (list of cluster counts per round).")

        client_data, testloader = build_mnist(n_clients, args.iid_alpha, args.batch_size, seed=args.seed)

        server_model = ConvNet()
        acc, loss, t = fl_semidecentralized_cluster_train(
            server_model, client_data, args.comm_rounds, args.lr, args.momentum, args.local_iters,
            args.straggler_max_delay, testloader, main_test_type="diff_num_of_clusters",
            num_clusters=cluster_schedule[0], cluster_schedule=cluster_schedule,
            reshuffle_clusters_each_round=getattr(args, "reshuffle_clusters_each_round", False),
            intra_graph_type=getattr(args, "intra_graph_type", "dense"), intra_graph_k=getattr(args, "intra_graph_k", 2),
            intra_graph_p=getattr(args, "intra_graph_p", 0.2), intra_graph_seed=args.seed,
            gossip_rounds=getattr(args, "gossip_rounds", 1), global_sync_E=getattr(args, "global_sync_E", 1),
        )

        plot_acc([acc], test_type, ["Dynamic clusters"], n_clients, "Accuracy (dynamic #clusters over rounds)")
        plot_loss([loss], test_type, ["Dynamic clusters"], n_clients, "Loss (dynamic #clusters over rounds)")
        plot_comm_times([t], test_type, ["Dynamic clusters"], n_clients, "Comm time (dynamic #clusters over rounds)")

    elif test_type.startswith("T8_partial_participation"):
        # Test Description:
        # Test different clients_frac (partial participation) and plot the results

        n_clients = getattr(args, "n_clients", 50)
        num_clusters = getattr(args, "num_clusters", 5)

        clients_frac_values = getattr(args, "clients_frac_values", [1.0, 0.5, 0.2, 0.1])

        intra_graph_type = getattr(args, "intra_graph_type", "dense")
        intra_graph_k = getattr(args, "intra_graph_k", 2)
        intra_graph_p = getattr(args, "intra_graph_p", 0.2)
        gossip_rounds = getattr(args, "gossip_rounds", 1)
        global_sync_E = getattr(args, "global_sync_E", 1)

        client_data, testloader = build_mnist(n_clients, args.iid_alpha, args.batch_size, seed=args.seed)

        accuracies, losses, times = [], [], []
        plot_labels = []

        for frac in clients_frac_values:
            server_model = ConvNet()
            acc, loss, t = fl_semidecentralized_cluster_train(
                server_model, client_data, args.comm_rounds, args.lr, args.momentum,
                args.local_iters, args.straggler_max_delay, testloader,
                main_test_type=test_type, num_clusters=num_clusters,
                intra_graph_type=intra_graph_type, intra_graph_k=intra_graph_k, intra_graph_p=intra_graph_p,
                intra_graph_seed=args.seed, gossip_rounds=gossip_rounds, global_sync_E=global_sync_E,
                clients_frac=frac
            )

            accuracies.append(acc)
            losses.append(loss)
            times.append(t)
            plot_labels.append(f"clients_frac={frac}")

        plot_acc(accuracies, test_type, plot_labels, n_clients, "Accuracy vs clients fraction")
        plot_loss(losses, test_type, plot_labels, n_clients, "Loss vs clients fraction")
        plot_comm_times(times, test_type, plot_labels, n_clients, "Comm time vs clients fraction")

    else:
        print("Unknown test type")
