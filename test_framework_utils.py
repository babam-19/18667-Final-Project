from models import ConvNet
from data_utils import build_mnist
from train_utils import fl_semidecentralized_cluster_train, fl_centralized_train, fl_fullydecentralized_cluster_train
from plot_utils import plot_acc, plot_loss, plot_comm_times

def test_cluster_framework_variations(test_type, args):
    
    if test_type.startswith("T1_eval_against_baselines"):
        # Test Description:
        # In this test, we will be evaluating our framework's performance against the baselines (being decentralized server 
        # architecture with local iters and two central server gossip network architectures being fully sync SGD and local 
        # iter SGD). We evaluate the accuracy, loss, and communication round time, and plot the results.

        n_clients = getattr(args, "n_clients", 50)
        num_clusters = getattr(args, "num_clusters", 5)
    
        # get all of the client data and the testloader
        client_data, testloader = build_mnist(n_clients, args.iid_alpha, args.batch_size, seed=args.seed)

        # ------------------------
        # creating the parameter server model for the fully decentralized arch with local iters
        decentralized_arch_server_model = ConvNet()
        # creating the parameter server model for the decentralized arch
        accuracies_decentralized_comm, losses_decentralized_comm, times_decentralized_comm = fl_fullydecentralized_cluster_train(decentralized_arch_server_model, client_data, args.comm_rounds, args.lr, args.momentum, args.local_iters, args.straggler_max_delay, testloader)
        # ------------------------

        # ------------------------
        # creating the parameter server model for the cluster arch with local iters
        cluster_arch_server_model = ConvNet()
        # our framework
        accuracies_cluster_comm, losses_cluster_comm, times_cluster_comm = fl_semidecentralized_cluster_train(cluster_arch_server_model, client_data, args.comm_rounds, args.lr, args.momentum, args.local_iters, args.straggler_max_delay, testloader, test_type, num_clusters=num_clusters)
        # ------------------------

        # ------------------------
        # creating the parameter server model for the central arch (fully sync SGD)
        fully_sync_SGD_server_model = ConvNet()
        # centralized fully sync SGD
        accuracies_fsync_SGD_comm, losses_fsync_SGD_comm, times_fsync_SGD_comm = fl_centralized_train(fully_sync_SGD_server_model, client_data, args.comm_rounds, args.lr, args.momentum, 1, args.straggler_max_delay, testloader)
        # ------------------------

        # ------------------------
        # creating the parameter server model for the central arch (local iter SGD)
        local_iter_server_model = ConvNet()
        # centralized local iter SGD
        accuracies_local_iter_comm, losses_local_iter_comm, times_local_iter_comm = fl_centralized_train(local_iter_server_model, client_data, args.comm_rounds, args.lr, args.momentum, args.local_iters, args.straggler_max_delay, testloader)
        # ------------------------

        plot_labels = ["Fully Decentralized Framework", "Our Semi-decentralized Framework", "Fully Sync SGD (centralized)", "Local Iter SGD (centralized)"]
        accuracies = [accuracies_decentralized_comm, accuracies_cluster_comm, accuracies_fsync_SGD_comm, accuracies_local_iter_comm]
        losses = [losses_decentralized_comm, losses_cluster_comm, losses_fsync_SGD_comm, losses_local_iter_comm]
        times = [times_decentralized_comm, times_cluster_comm, times_fsync_SGD_comm, times_local_iter_comm]
        
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
        # creating the parameter server model for the central arch (fully sync SGD)
        fully_sync_SGD_server_model = ConvNet()
        # centralized fully sync SGD
        accuracies_fsync_SGD_comm, losses_fsync_SGD_comm, times_fsync_SGD_comm = fl_centralized_train(fully_sync_SGD_server_model, client_data, args.comm_rounds, args.lr, args.momentum, 1, args.straggler_max_delay, testloader)
        # ------------------------

        # ------------------------
        # creating the parameter server model for the central arch (local iter SGD)
        local_iter_server_model = ConvNet()
        # centralized local iter SGD
        accuracies_local_iter_comm, losses_local_iter_comm, times_local_iter_comm = fl_centralized_train(local_iter_server_model, client_data, args.comm_rounds, args.lr, args.momentum, args.local_iters, args.straggler_max_delay, testloader)
        # ------------------------

        plot_labels = [f"Our Semi-decentralized Framework (clusters = {c})" for c in cluster_counts]
        plot_labels.extend(["Fully Sync SGD (centralized)", "Local Iter SGD (centralized)"])
        accuracies.extend([accuracies_fsync_SGD_comm, accuracies_local_iter_comm])
        losses.extend([losses_fsync_SGD_comm, losses_local_iter_comm])
        times.extend([times_fsync_SGD_comm, times_local_iter_comm])
        
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
        # creating the parameter server model for the central arch (fully sync SGD)
        fully_sync_SGD_server_model = ConvNet()
        # centralized fully sync SGD
        accuracies_fsync_SGD_comm, losses_fsync_SGD_comm, times_fsync_SGD_comm = fl_centralized_train(fully_sync_SGD_server_model, client_data, args.comm_rounds, args.lr, args.momentum, 1, args.straggler_max_delay, testloader)
        # ------------------------

        # ------------------------
        # creating the parameter server model for the central arch (local iter SGD)
        local_iter_server_model = ConvNet()
        # centralized local iter SGD
        accuracies_local_iter_comm, losses_local_iter_comm, times_local_iter_comm = fl_centralized_train(local_iter_server_model, client_data, args.comm_rounds, args.lr, args.momentum, args.local_iters, args.straggler_max_delay, testloader)
        # ------------------------

        plot_labels = ["Our Semi-decentralized Framework (clusters = 10)", "Fully Sync SGD (centralized)", "Local Iter SGD (centralized)"]
        accuracies = [accuracies_cluster_comm, accuracies_fsync_SGD_comm, accuracies_local_iter_comm]
        losses = [losses_cluster_comm, losses_fsync_SGD_comm, losses_local_iter_comm]
        times = [times_cluster_comm, times_fsync_SGD_comm, times_local_iter_comm]
        
        # plot everything
        plot_acc(accuracies, test_type, plot_labels, n_clients, 'Accuracy w/ scaled up num of clients')
        plot_loss(losses, test_type, plot_labels, n_clients, 'Loss w/ scaled up num of clients')
        plot_comm_times(times, test_type, plot_labels, n_clients, 'Communication Times w/ scaled up num of clients')

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
        # Fixed number of total clients, but uneven cluster sizes vs equal-split baseline (same #clusters)
        n_clients = getattr(args, "n_clients", 50)
        num_clusters = getattr(args, "num_clusters", 5)
        cluster_sizes = getattr(args, "cluster_sizes", None)
        if cluster_sizes is None:
            raise ValueError("T6 requires args.cluster_sizes (list of cluster sizes that sums to n_clients).")

        client_data, testloader = build_mnist(n_clients, args.iid_alpha, args.batch_size, seed=args.seed)

        intra_graph_type = getattr(args, "intra_graph_type", "dense")
        intra_graph_k = getattr(args, "intra_graph_k", 2)
        intra_graph_p = getattr(args, "intra_graph_p", 0.2)
        gossip_rounds = getattr(args, "gossip_rounds", 1)
        global_sync_E = getattr(args, "global_sync_E", 1)

        # Uneven cluster sizes
        server_model_uneven = ConvNet()
        acc_uneven, loss_uneven, t_uneven = fl_semidecentralized_cluster_train(
            server_model_uneven, client_data, args.comm_rounds, args.lr, args.momentum, args.local_iters,
            args.straggler_max_delay, testloader,
            main_test_type="diff_num_of_clients_per_cluster",
            num_clusters=num_clusters,
            cluster_sizes=cluster_sizes,
            intra_graph_type=intra_graph_type, intra_graph_k=intra_graph_k, intra_graph_p=intra_graph_p,
            intra_graph_seed=args.seed,
            gossip_rounds=gossip_rounds,
            global_sync_E=global_sync_E,
            clients_frac=getattr(args, "clients_frac", 1.0),
        )

        # (baseline): Equal split with same num_clusters
        server_model_equal = ConvNet()
        acc_equal, loss_equal, t_equal = fl_semidecentralized_cluster_train(
            server_model_equal, client_data, args.comm_rounds, args.lr, args.momentum, args.local_iters,
            args.straggler_max_delay, testloader,
            main_test_type="T1_eval_against_baselines",   # any eq-split tag that routes to equal split
            num_clusters=num_clusters,
            cluster_sizes=None,
            intra_graph_type=intra_graph_type, intra_graph_k=intra_graph_k, intra_graph_p=intra_graph_p,
            intra_graph_seed=args.seed,
            gossip_rounds=gossip_rounds,
            global_sync_E=global_sync_E,
            clients_frac=getattr(args, "clients_frac", 1.0),
        )

        plot_labels = [f"Uneven clusters {cluster_sizes}", f"Baseline: equal split ({num_clusters} clusters)"]
        plot_acc([acc_uneven, acc_equal], test_type, plot_labels, n_clients, "Accuracy (uneven vs equal cluster sizes)")
        plot_loss([loss_uneven, loss_equal], test_type, plot_labels, n_clients, "Loss (uneven vs equal cluster sizes)")
        plot_comm_times([t_uneven, t_equal], test_type, plot_labels, n_clients, "Comm time (uneven vs equal cluster sizes)")

    elif test_type.startswith("T7_diff_num_of_clusters_over_rounds"):
        # Test Description:
        # Change number of clusters over rounds using a schedule vs fixed-#clusters baseline.
        n_clients = getattr(args, "n_clients", 50)
        cluster_schedule = getattr(args, "cluster_schedule", None)
        if cluster_schedule is None:
            raise ValueError("T7 requires args.cluster_schedule (list of cluster counts per round).")
        if len(cluster_schedule) < args.comm_rounds:
            raise ValueError(f"T7: cluster_schedule length {len(cluster_schedule)} < comm_rounds {args.comm_rounds}")

        # choose baseline cluster count (fixed across rounds)
        baseline_num_clusters = getattr(args, "baseline_num_clusters", int(cluster_schedule[0]))

        client_data, testloader = build_mnist(n_clients, args.iid_alpha, args.batch_size, seed=args.seed)

        intra_graph_type = getattr(args, "intra_graph_type", "dense")
        intra_graph_k = getattr(args, "intra_graph_k", 2)
        intra_graph_p = getattr(args, "intra_graph_p", 0.2)
        gossip_rounds = getattr(args, "gossip_rounds", 1)
        global_sync_E = getattr(args, "global_sync_E", 1)
        reshuffle = getattr(args, "reshuffle_clusters_each_round", False)

        # Dynamic clusters using schedule
        server_model_dynamic = ConvNet()
        acc_dyn, loss_dyn, t_dyn = fl_semidecentralized_cluster_train(
            server_model_dynamic, client_data, args.comm_rounds, args.lr, args.momentum, args.local_iters,
            args.straggler_max_delay, testloader,
            main_test_type="diff_num_of_clusters",
            num_clusters=int(cluster_schedule[0]),
            cluster_schedule=cluster_schedule,
            reshuffle_clusters_each_round=reshuffle,
            intra_graph_type=intra_graph_type, intra_graph_k=intra_graph_k, intra_graph_p=intra_graph_p,
            intra_graph_seed=args.seed,
            gossip_rounds=gossip_rounds,
            global_sync_E=global_sync_E,
            clients_frac=getattr(args, "clients_frac", 1.0),
        )

        # (baseline): Fixed clusters across all rounds
        server_model_fixed = ConvNet()
        acc_fix, loss_fix, t_fix = fl_semidecentralized_cluster_train(
            server_model_fixed, client_data, args.comm_rounds, args.lr, args.momentum, args.local_iters,
            args.straggler_max_delay, testloader,
            main_test_type="T1_eval_against_baselines",  # routes to equal split behavior
            num_clusters=baseline_num_clusters,
            cluster_schedule=None,
            reshuffle_clusters_each_round=reshuffle,
            intra_graph_type=intra_graph_type, intra_graph_k=intra_graph_k, intra_graph_p=intra_graph_p,
            intra_graph_seed=args.seed,
            gossip_rounds=gossip_rounds,
            global_sync_E=global_sync_E,
            clients_frac=getattr(args, "clients_frac", 1.0),
        )

        plot_labels = [f"Dynamic clusters (schedule)", f"Baseline: fixed clusters = {baseline_num_clusters}"]
        plot_acc([acc_dyn, acc_fix], test_type, plot_labels, n_clients, "Accuracy (dynamic vs fixed #clusters)")
        plot_loss([loss_dyn, loss_fix], test_type, plot_labels, n_clients, "Loss (dynamic vs fixed #clusters)")
        plot_comm_times([t_dyn, t_fix], test_type, plot_labels, n_clients, "Comm time (dynamic vs fixed #clusters)")

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
