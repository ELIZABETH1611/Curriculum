import torch
import torch_scatter
from typing import Union
# Loads the openmp shared libraries (current bug of the library)
import sklearn
from pykeops.torch import LazyTensor, Vj


def euclidean_low_mem_assignment(x: torch.Tensor, y: torch.Tensor, epsilon: float, max_unassigned: int = 0,
                                 max_iterations: Union[int, str] = "auto", debug: bool = False):
    # For PyKeops, it is better to have contiguous memory (this call is free if the tensors are already contiguous)
    x = x.contiguous()
    y = y.contiguous()

    # We always assume to have at most as many buyers (first dimension) as objects to be sold
    transposed = x.shape[0] > y.shape[0]
    if transposed:
        tmp = x
        x = y
        y = tmp

    # We create the lazy tensors around x and y
    lt_x = LazyTensor(x[:, None, :])
    lt_y = LazyTensor(y[None, :, :])
    squared_dist = ((lt_x - lt_y) ** 2).sum(-1)
    max_squared_dist = squared_dist.max(dim=1).max(dim=0).values
    min_squared_dist = squared_dist.min(dim=1).min(dim=0).values
    benefits = max_squared_dist - squared_dist

    if max_iterations == "auto":
        max_iterations = torch.ceil(torch.tensor(1 / epsilon, dtype=x.dtype, device=x.device)).long()

    epsilon = epsilon * (max_squared_dist - min_squared_dist)

    if debug:
        ref_benefits = torch.sum(torch.square(x[:, None, :] - y[None, :, :]), dim=-1)
        ref_benefits = torch.max(ref_benefits) - ref_benefits
        assert torch.allclose(ref_benefits.max(dim=0).values, benefits.max(dim=0)[:, 0])
        assert torch.allclose(ref_benefits.min(dim=0).values, benefits.min(dim=0)[:, 0])
        assert torch.allclose(ref_benefits.min(dim=0).values, benefits.min(dim=0)[:, 0])
        assert torch.allclose(ref_benefits.max(dim=1).values, benefits.max(dim=1)[:, 0])
        assert torch.allclose(ref_benefits.min(dim=1).values, benefits.min(dim=1)[:, 0])

    n_buyers, n_objects = x.shape[0], y.shape[0]
    prices = torch.zeros(n_objects, device=x.device, dtype=x.dtype)
    # An assignment value of -1 means unassigned
    buyer_assignments = torch.zeros(n_buyers, device=x.device, dtype=torch.int64) - 1
    object_assignments = torch.zeros(n_objects, device=x.device, dtype=torch.int64) - 1

    unassigned_buyers = torch.where(buyer_assignments == -1)[0]
    iteration_count = 0
    while len(unassigned_buyers) > max_unassigned and iteration_count < max_iterations:
        # Bidding Phase:
        # 1. Get the object values for those buyers that are not yet assigned
        lt_x = LazyTensor(x[unassigned_buyers, None, :])
        lt_y = LazyTensor(y[None, :, :])
        squared_dist = ((lt_x - lt_y) ** 2).sum(-1)
        benefits = max_squared_dist - squared_dist
        # Interestingly, this operation produced error for operations with a large number of dimensions and a only one unassigned buyer
        # Disabling chunking prevents this error (while it should not significantly affect the performance)
        try:
            top_values, top_idxs = (-(benefits - Vj(prices[:, None]))).Kmin_argKmin(2, dim=1, enable_chunks=
            unassigned_buyers.shape[0] > 1)
        except ValueError as e:
            top_values, top_idxs = (-(benefits - Vj(prices[:, None]))).Kmin_argKmin(2, dim=1, enable_chunks=False)
        top_values = -1 * top_values
        top_idxs = top_idxs

        if debug:
            ref_top_values, ref_top_idxs = (ref_benefits[unassigned_buyers, :] - prices[None, :]).topk(2, dim=1)
            assert torch.all(ref_top_idxs == top_idxs)
            assert torch.allclose(ref_top_values, top_values)

        # 2. Compute the bids of each unassigned buyer for their most-valuable object
        bids = prices[top_idxs[:, 0]] + top_values[:, 0] - top_values[:, 1] + epsilon

        # Assignment Phase:
        # For each object get the number of bidders
        objects_with_bids, bid_assigment = torch.unique(top_idxs[:, 0], return_inverse=True)

        # Highest bids for the objects indexed as objects[objects_with_bids]. The highest bidder tensor hold
        # the indices of the highest bidding unassigned person
        highest_bids, highest_bidder = torch_scatter.scatter_max(bids, bid_assigment)

        # We update the prices
        prices[objects_with_bids] = highest_bids

        # Finally, we need to update the assignments by first removing the object from the old buyer and adding it
        # to the new buyers
        old_buyers = object_assignments[objects_with_bids]
        # We need to ignore those old buyers that have the value "-1", meaning unassigned
        old_buyers = old_buyers[old_buyers != -1]
        object_assignments[buyer_assignments[old_buyers]] = -1
        buyer_assignments[old_buyers] = -1

        new_buyers = unassigned_buyers[highest_bidder]
        buyer_assignments[new_buyers] = objects_with_bids
        object_assignments[objects_with_bids] = new_buyers

        # Re-Compute the unassigned buyers for the next interation
        unassigned_buyers = torch.where(buyer_assignments == -1)[0]
        iteration_count += 1

    # In case we terminated early, we simply randomly assign the best M unassigned objects among the M unassigned buyers
    n_unassigned = len(unassigned_buyers)
    unassigned_objects = torch.where(object_assignments == -1)[0]

    # Quick sanity check that nothing went terribly wrong
    assert n_buyers - n_unassigned == n_objects - unassigned_objects.shape[0]

    if n_unassigned > 0:
        buyer_assignments[unassigned_buyers] = \
            unassigned_objects[torch.randperm(unassigned_objects.shape[0], device=x.device)[:n_unassigned]]

    if transposed:
        return buyer_assignments, torch.arange(n_buyers, device=buyer_assignments.device)
    else:
        return torch.arange(n_buyers, device=buyer_assignments.device), buyer_assignments


def auction(benefits: torch.Tensor, epsilon: float, max_iterations: Union[int, str] = "auto",
            max_unassigned: int = 0):
    # We always assume to have at most as many buyers (first dimension) as objects to be sold
    transposed = benefits.shape[0] > benefits.shape[1]
    if transposed:
        benefits = benefits.T

    if max_iterations == "auto":
        max_iterations = torch.ceil(torch.max(benefits) / epsilon).long()

    n_buyers, n_objects = benefits.shape
    profits = torch.zeros(n_buyers, device=benefits.device)
    prices = torch.zeros(n_objects, device=benefits.device)
    # An assignment value of -1 means unassigned
    buyer_assignments = torch.zeros(n_buyers, device=benefits.device, dtype=torch.int64) - 1
    object_assignments = torch.zeros(n_objects, device=benefits.device, dtype=torch.int64) - 1
    lmbd = 0.

    # Only Forward for now (does not use profits and lmbd)
    unassigned_buyers = torch.where(buyer_assignments == -1)[0]
    iteration_count = 0
    while len(unassigned_buyers) > max_unassigned and iteration_count < max_iterations:
        # Bidding Phase:
        # 1. Get the object values for those buyers that are not yet assigned
        values = benefits[unassigned_buyers, :] - prices[None, :]
        # 2. Compute the bids of each unassigned buyer for their most-valuable object
        top_values, top_idxs = values.topk(2, dim=1)
        bids = prices[top_idxs[:, 0]] + top_values[:, 0] - top_values[:, 1] + epsilon

        # Assignment Phase:
        # For each object get the number of bidders
        objects_with_bids, bid_assigment = torch.unique(top_idxs[:, 0], return_inverse=True)

        # Highest bids for the objects indexed as objects[objects_with_bids]. The highest bidder tensor hold
        # the indices of the highest bidding unassigned person
        highest_bids, highest_bidder = torch_scatter.scatter_max(bids, bid_assigment)

        # We update the prices
        prices[objects_with_bids] = highest_bids

        # Finally, we need to update the assignments by first removing the object from the old buyer and adding it
        # to the new buyers
        old_buyers = object_assignments[objects_with_bids]
        # We need to ignore those old buyers that have the value "-1", meaning unassigned
        old_buyers = old_buyers[old_buyers != -1]
        object_assignments[buyer_assignments[old_buyers]] = -1
        buyer_assignments[old_buyers] = -1

        new_buyers = unassigned_buyers[highest_bidder]
        buyer_assignments[new_buyers] = objects_with_bids
        object_assignments[objects_with_bids] = new_buyers

        # Re-Compute the unassigned buyers for the next interation
        unassigned_buyers = torch.where(buyer_assignments == -1)[0]
        iteration_count += 1

    # In case we terminated early, we simply randomly assign the best M unassigned objects among the M unassigned buyers
    n_unassigned = len(unassigned_buyers)
    unassigned_objects = torch.where(object_assignments == -1)[0]

    # Quick sanity check that nothing went terribly wrong
    assert n_buyers - n_unassigned == n_objects - unassigned_objects.shape[0]

    if n_unassigned > 0:
        # This step only makes a difference for assymetric assignment problems
        top_idxs = torch.unique(
            torch.topk(benefits[unassigned_buyers[:, None], unassigned_objects[None, :]], k=n_unassigned)[1])
        buyer_assignments[unassigned_buyers] = unassigned_objects[
            [top_idxs[torch.randperm(n_unassigned, device=benefits.device)]]]

    if transposed:
        return buyer_assignments, torch.arange(n_buyers)
    else:
        return torch.arange(n_buyers), buyer_assignments


def assignment(costs: torch.Tensor, epsilon: float, max_iterations: Union[int, str] = "auto",
               max_unassigned: int = 0):
    # We transform the costs into benefits and ensure that they are lower bounded by zero
    return auction(torch.max(costs) - costs, epsilon, max_iterations=max_iterations, max_unassigned=max_unassigned)


if __name__ == "__main__":
    import time
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    n1 = 2 * 8192
    # n2 = 2 * 8192 + 2048
    n2 = 2048
    dim = 60

    np.random.seed(4)

    x = np.random.uniform(-1, 1, size=(n1, dim))
    # x /= np.linalg.norm(x, axis=1, keepdims=True)
    # x *= 0.03

    y = np.random.uniform(-1, 1, size=(n2, dim))
    # y = (x + np.random.uniform(-0.01, 0.01, size=(n1, dim)))[np.random.permutation(n1)]
    # n2 = n1

    x_torch = torch.from_numpy(x).to("cuda:0").float()
    y_torch = torch.from_numpy(y).to("cuda:0").float()
    if n1 <= 4096:
        distances_np = np.linalg.norm(x[:, None, :] - y[None, ...], axis=-1) ** 2
        distances = torch.from_numpy(distances_np).float().to("cuda:0")

        print(f"Random Benefit: {np.mean(distances_np[np.arange(n1), np.random.permutation(n2)[:n1]])}")

        t1 = time.time()
        for i in range(10):
            res1 = assignment(distances, epsilon=0.02 * (torch.max(distances) - torch.min(distances)),
                              max_iterations=50)
            assert torch.unique(res1[0]).shape[0] == min(n1, n2) and torch.unique(res1[1]).shape[0] == min(n1, n2)
            res2 = assignment(distances.T, epsilon=0.02 * (torch.max(distances) - torch.min(distances)),
                              max_iterations=50)
            assert torch.unique(res1[0]).shape[0] == min(n1, n2) and torch.unique(res1[1]).shape[0] == min(n1, n2)
        t2 = time.time()
        print(f"Benefit: {torch.max(torch.mean(distances[res1[0], res1[1]]), torch.mean(distances[res2[1], res2[0]]))}")
        print(f"GPU Auction took {(t2 - t1) / 20: .4f} seconds")
    else:
        print("Skipping regular auction because the matrices are too big")

    t1 = time.time()
    for i in range(10):
        res1 = euclidean_low_mem_assignment(x_torch, y_torch, epsilon=0.02)
        assert torch.unique(res1[0]).shape[0] == min(n1, n2) and torch.unique(res1[1]).shape[0] == min(n1, n2)
        res2 = euclidean_low_mem_assignment(y_torch, x_torch, epsilon=0.02)
        assert torch.unique(res1[0]).shape[0] == min(n1, n2) and torch.unique(res1[1]).shape[0] == min(n1, n2)
    t2 = time.time()
    print(f"Benefit: {torch.mean(torch.sum(torch.square(x_torch[res1[0]] - y_torch[res1[1]]), dim=-1))}, \
          {torch.mean(torch.sum(torch.square(x_torch[res2[1]] - y_torch[res2[0]]), axis=-1))}")
    print(f"Low-Mem GPU Auction took {(t2 - t1) / 20: .4f} seconds")

    if n1 <= 4096:
        t1 = time.time()
        for i in range(2):
            rows, cols = linear_sum_assignment(distances_np, maximize=False)
        t2 = time.time()
        print(f"Scipy LSA took {(t2 - t1) / 2: .4f} seconds")
        print(f"Scipy-Benefit: {np.mean(distances_np[rows, cols])}")

        t1 = time.time()
        for i in range(2):
            rows, cols = linear_sum_assignment(distances_np.T, maximize=False)
        t2 = time.time()
        print(f"Scipy LSA-Transposed took {(t2 - t1) / 2: .4f} seconds")
        print(f"Scipy-Benefit: {np.mean(distances_np[cols, rows])}")
    else:
        print("Skipping SciPy LSA because the matrices are too big")
