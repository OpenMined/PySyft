import torch


def _generate_mpc_number_pair(self, n1, n2):
    mpcs = []
    for i in [n1, n2]:
        mpcs.append(torch.LongTensor([i]).share(self.bob, self.alice))

    return mpcs
