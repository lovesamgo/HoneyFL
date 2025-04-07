import time
import random


def generate_mapping(args, seed):
    random.seed(seed)
    if args.dataset == 'mnist':
        num_classes = 10
    elif args.dataset == 'cifar':
        num_classes = 10
    elif args.dataset == 'gtsrb':
        num_classes = 43
    elif args.dataset == 'imagenet':
        num_classes = 100
    else:
        raise ValueError(f"Invalid dataset: {args.dataset}")
    digits = list(range(num_classes))
    # Sattolo算法
    n = len(digits)
    while n > 1:
        n -= 1
        j = random.randrange(n)  # 0 <= j < n
        digits[j], digits[n] = digits[n], digits[j]
    return digits


if __name__ == '__main__':
    interval = 30  # 30秒
    current_seed = int(time.time()) // interval
    mapping = generate_mapping(current_seed)
    print(mapping)