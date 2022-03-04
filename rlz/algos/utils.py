from copy import deepcopy


def make_target_network(origin):
    target = deepcopy(origin)
    for param in target.parameters():
        param.requires_grad_(False)
    return target


def polyak_average(origin, target, tau):
    for origin_param, target_param in zip(origin.parameters(), target.parameters()):
        target_param.data.mul_(1 - tau).add_(origin_param, alpha=tau)
