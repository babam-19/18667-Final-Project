from models import ConvNet
from data_utils import build_mnist
from train_utils import fl_semidecentralized_cluster_train, fl_centralized_train, fl_fullydecentralized_cluster_train
from plot_utils import plot_acc, plot_loss, plot_comm_times

def test_cluster_framework_variations(test_type, args):
    
    if test_type == "T1_eval_against_baselines":

        # Test Description:
        # In this test, we will be evaluating our framework's performance against the baselines (being decentralized server 
        # architecture with local iters and two central server gossip network architectures being fully sync SGD and local 
        # iter SGD). We evaluate the accuracy, loss, and communication round time, and plot the results.

        n_clients = 50
    
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
        accuracies_cluster_comm, losses_cluster_comm, times_cluster_comm = fl_semidecentralized_cluster_train(cluster_arch_server_model, client_data, args.comm_rounds, args.lr, args.momentum, args.local_iters, args.straggler_max_delay, testloader, "T1_eval_against_baselines", num_clusters=5)
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

        plot_labels = ["Our Semi-decentralized Framework (clusters = 5)", "Our Semi-decentralized Framework (clusters = 10)",  "Our Semi-decentralized Framework (clusters = 25)", "Fully Sync SGD (centralized)", "Local Iter SGD (centralized)"]
        accuracies = [accuracies_cluster_comm_5, accuracies_cluster_comm_10, accuracies_cluster_comm_25, accuracies_fsync_SGD_comm, accuracies_local_iter_comm]
        losses = [losses_cluster_comm_5, losses_cluster_comm_10, losses_cluster_comm_25, losses_fsync_SGD_comm, losses_local_iter_comm]
        times = [times_cluster_comm_5, times_cluster_comm_10, times_cluster_comm_25, times_fsync_SGD_comm, times_local_iter_comm]
        
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
        
    else:
        print("Unknown test type")
