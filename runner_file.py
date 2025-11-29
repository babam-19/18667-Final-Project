import argparse
from argparse import RawTextHelpFormatter
from test_framework_utils import test_cluster_framework_variations


def main(args): 
    print("Beginning Tests")

    # TESTING OUR FRAMEWORK WITH VARIATIONS OF THE ARCHITECTURE

    tests_to_run = args.test_type

    if "T1_eval_against_baselines" in tests_to_run or "all" in tests_to_run:
        # test our framework against the other baselines
        print("TEST 1: Testing when there are an equal amount of clients in a cluster vs the baseline fully sync SGD and decentralized FL")
        test_cluster_framework_variations("T1_eval_against_baselines", args)
        print("Done with TEST 1")
        print("---------------------------------")

    if "T2_change_num_of_clusters" in tests_to_run or "all" in tests_to_run:
        # test our framework with adapted num of clusters against one of the baselines
        print("TEST 2: Testing when the number of clusters are adapted (changing the num of clusters) vs the baseline fully sync SGD")
        test_cluster_framework_variations("T2_change_num_of_clusters", args)
        print("Done with TEST 2")
        print("---------------------------------")

    if "T3_scale_up_num_of_clients" in tests_to_run or "all" in tests_to_run:
        # test our framework with scaled up num of clients against one of the baselines
        print("TEST 3: Testing when we scale up the number of clients vs the baseline fully sync SGD")
        test_cluster_framework_variations("T3_scale_up_num_of_clients", args)
        print("Done with TEST 3")
        print("---------------------------------")




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
    
    # Isolating tests param
    parser.add_argument('--test_type', nargs='*', type=str, default='all',
        help='What test you want to run')
    
    args = parser.parse_args()

    main(args)