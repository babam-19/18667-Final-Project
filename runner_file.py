import argparse
from copy import deepcopy
from test_framework_utils import test_cluster_framework_variations

def base_args():
    return argparse.Namespace(
        seed=123,
        batch_size=32,
        comm_rounds=30,
        lr=1e-3,
        momentum=0.9,
        local_iters=10,
        straggler_max_delay=0,
        iid_alpha=-1,
        clients_frac=1.0,
        global_sync_E=1,
        gossip_rounds=1,
        intra_graph_type="dense",
        intra_graph_k=2,
        intra_graph_p=0.2,
        cluster_sizes=None,
        cluster_schedule=None,
        reshuffle_clusters_each_round=False,
    )

def main():
    tests = [
        # # T1: test our framework against the other baselines
        # ("T1_eval_against_baselines__iid", dict(n_clients=50, iid_alpha=-1, num_clusters=5)),
        # ("T1_eval_against_baselines__non_iid_a0p3", dict(n_clients=50, iid_alpha=0.3, num_clusters=5)),
        # ("T1_eval_against_baselines__non_iid_a0p1", dict(n_clients=50, iid_alpha=0.1, num_clusters=5)),

        #  # T2: test our framework with adapted num of clusters against one of the baselines
        # ("T2_change_num_of_clusters__iid", dict(n_clients=50, cluster_counts=[5,10,25])),
        # ("T2_change_num_of_clusters__non_iid_a0p3", dict(n_clients=50, iid_alpha=0.3, cluster_counts=[5,10,25])),

        # # T3: test our framework with scaled up num of clients against one of the baselines
        # ("T3_scale_up_num_of_clients__iid", dict(n_clients=150, iid_alpha=-1, num_clusters=10)),
        # ("T3_scale_up_num_of_clients__non_iid_a0p3", dict(n_clients=150, iid_alpha=0.3, num_clusters=10)),

        # # T4: test our framework with different graph connectivities
        # ("T4_graph_connectivity", dict(
        #     n_clients=50,
        #     num_clusters=5,
        #     iid_alpha=0.3,

        #     connectivity_configs=[
        #         {"label": "dense", "graph_type": "dense", "k": 2, "p": 0.2, "gossip_rounds": 5},
        #         {"label": "ring(k=2)", "graph_type": "ring", "k": 2, "p": 0.2, "gossip_rounds": 5},
        #         {"label": "ring(k=4)", "graph_type": "ring", "k": 4, "p": 0.2, "gossip_rounds": 5},
        #         {"label": "random(p=0.2)", "graph_type": "random", "k": 2, "p": 0.2, "gossip_rounds": 5},
        #         {"label": "star", "graph_type": "star", "k": 2, "p": 0.2, "gossip_rounds": 5},
        #     ],
        # )),

        # # T5: test different global sync frequencies E (sync every E rounds)
        # ("T5_global_sync_frequency__non_iid_a0p3", dict(
        #     n_clients=50,
        #     num_clusters=5,
        #     iid_alpha=0.3,
        #     intra_graph_type="dense",
        #     intra_graph_k=2,
        #     intra_graph_p=0.2,
        #     gossip_rounds=1,
        #     E_values=[1, 2, 5, 10],
        # )),

        # T6: different number of clients per cluster (fixed uneven cluster sizes)
        ("T6_diff_num_of_clients_per_cluster__non_iid_a0p3", dict(
            n_clients=50,
            iid_alpha=0.3,
            num_clusters=5,
            cluster_sizes=[5, 8, 10, 12, 15],
            intra_graph_type="dense",
            intra_graph_k=2,
            intra_graph_p=0.2,
            gossip_rounds=1,
            global_sync_E=5,
        )),

        # T7: different number of clusters per round (dynamic schedule)
        ("T7_diff_num_of_clusters_over_rounds__non_iid_a0p3", dict(
            n_clients=50,
            iid_alpha=0.3,
            cluster_schedule=[5]*10 + [10]*10 + [25]*10,
            intra_graph_type="dense",
            intra_graph_k=2,
            intra_graph_p=0.2,
            gossip_rounds=1,
            global_sync_E=5,
            reshuffle_clusters_each_round=True,
        )),

        # # T8: partial participation
        # ("T8_partial_participation__non_iid_a0p3", dict(
        #     n_clients=50,
        #     num_clusters=5,
        #     iid_alpha=0.3,
        #     intra_graph_type="dense",
        #     intra_graph_k=2,
        #     intra_graph_p=0.2,
        #     gossip_rounds=1,
        #     global_sync_E=5,
        #     clients_frac_values=[1.0, 0.5, 0.2, 0.1],
        # )),

    ]
    print("Beginning Tests")
    for test_name, overrides in tests:
        args = deepcopy(base_args())
        for k, v in overrides.items():
            setattr(args, k, v)

        print(f"\n===== RUNNING: {test_name} =====")
        test_cluster_framework_variations(test_name, args)

if __name__ == "__main__":
    main()
