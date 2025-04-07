import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    # federated arguments
    parser.add_argument('--epochs', type=int, default=50, help="rounds of training")
    parser.add_argument('--num_users', type=int, default=14, help="number of users: K")
    parser.add_argument('--frac', type=float, default=0.75, help="the fraction of clients: C")
    parser.add_argument('--local_ep', type=int, default=5, help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=20, help="local batch size: B")
    parser.add_argument('--bs', type=int, default=128, help="test batch size")
    parser.add_argument('--lr', type=float, default=0.1, help="learning rate")
    parser.add_argument('--momentum', type=float, default=0.5, help="SGD momentum (default: 0.5)")

    # other arguments
    parser.add_argument('--dataset', type=str, default='cifar', help="name of dataset")
    parser.add_argument('--poison_client_portion', type=float, default=0.2, help="the number of poisoned clients portion")
    parser.add_argument('--honeypot_client_portion', type=float, default=0.2, help="the number of honeypot portion")
    parser.add_argument('--poison_portion', type=float, default=0.5, help="portion of poisoned data")
    parser.add_argument('--honeypot_portion', type=float, default=0.5, help="portion of honeypot data")
    parser.add_argument('--trigger_label', type=int, default=0, help="label of poison trigger")
    parser.add_argument('--iid', action='store_true', help='whether i.i.d or not')
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    parser.add_argument('--gpu', type=int, default=1, help="GPU ID, -1 for CPU")
    parser.add_argument('--verbose', action='store_true', help='verbose print')
    parser.add_argument('--all_clients', action='store_true', help='aggregation over all clients')
    args = parser.parse_args()
    return args
