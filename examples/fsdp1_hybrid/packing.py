import heapq
import os
from functools import partial
from typing import List

import numpy as np
import torch
from args import get_args
from torch import distributed as dist


def bin_packing_first_fit_decreasing(bin_capacity, items):
    """
    Solve the bin packing problem using First Fit Decreasing algorithm

    Args:
        bin_capacity (int): Capacity of each bin
        items (list): List of item volumes

    Returns:
        tuple: (minimum number of bins, bin allocation details (indices), bin allocation details (volumes))
    """
    if not items:
        return 0, [], []

    # Create a list of (volume, original_index) tuples, sorted by volume in descending order
    indexed_items = [(volume, idx) for idx, volume in enumerate(items)]
    indexed_items.sort(key=lambda x: x[0], reverse=True)

    # Initialize bin lists
    bins_volume = []  # Store volumes
    bins_index = []  # Store indices

    for volume, original_idx in indexed_items:
        # Try to place the item into an existing bin
        placed = False
        for i, bin_content in enumerate(bins_volume):
            if sum(bin_content) + volume <= bin_capacity:
                bins_volume[i].append(volume)
                bins_index[i].append(original_idx)
                placed = True
                break

        # If unable to fit into existing bins, create a new bin
        if not placed:
            bins_volume.append([volume])
            bins_index.append([original_idx])

    return len(bins_volume), bins_index, bins_volume


def bin_packing_best_fit_decreasing(bin_capacity, items):
    """
    Solve the bin packing problem using Best Fit Decreasing algorithm

    Args:
        bin_capacity (int): Capacity of each bin
        items (list): List of item volumes

    Returns:
        tuple: (minimum number of bins, bin allocation details (indices), bin allocation details (volumes))
    """
    if not items:
        return 0, [], []

    # Create a list of (volume, original_index) tuples, sorted by volume in descending order
    indexed_items = [(volume, idx) for idx, volume in enumerate(items)]
    indexed_items.sort(key=lambda x: x[0], reverse=True)

    # Initialize bin lists
    bins_volume = []  # Store volumes
    bins_index = []  # Store indices

    for volume, original_idx in indexed_items:
        # Find the bin with the smallest remaining space that can fit this item
        best_bin_idx = -1
        min_remaining_space = float("inf")

        for i, bin_content in enumerate(bins_volume):
            remaining_space = bin_capacity - sum(bin_content)
            if remaining_space >= volume and remaining_space < min_remaining_space:
                min_remaining_space = remaining_space
                best_bin_idx = i

        # If a suitable bin is found, place the item; otherwise create a new bin
        if best_bin_idx != -1:
            bins_volume[best_bin_idx].append(volume)
            bins_index[best_bin_idx].append(original_idx)
        else:
            bins_volume.append([volume])
            bins_index.append([original_idx])

    return len(bins_volume), bins_index, bins_volume


def karmarkar_karp(seq_cost_list: list[int], k_partitions: int, equal_size: bool):
    # see: https://en.wikipedia.org/wiki/Largest_differencing_method
    class Set:
        def __init__(self) -> None:
            self.sum = 0
            self.items = []

        def add(self, idx: int, val: int):
            self.items.append((idx, val))
            self.sum += val

        def merge(self, other):
            for idx, val in other.items:
                self.items.append((idx, val))
                self.sum += val

        def __lt__(self, other):
            if self.sum != other.sum:
                return self.sum < other.sum
            if len(self.items) != len(other.items):
                return len(self.items) < len(other.items)
            return self.items < other.items

    class State:
        def __init__(self, items: list[tuple[int, int]], k: int) -> None:
            self.k = k
            # sets should always be decreasing order
            self.sets = [Set() for _ in range(k)]
            assert len(items) in [1, k], f"{len(items)} not in [1, {k}]"
            for i, (idx, seqlen) in enumerate(items):
                self.sets[i].add(idx=idx, val=seqlen)
            self.sets = sorted(self.sets, reverse=True)

        def get_partitions(self):
            partitions = []
            for i in range(len(self.sets)):
                cur_partition = []
                for idx, _ in self.sets[i].items:
                    cur_partition.append(idx)
                partitions.append(cur_partition)
            return partitions

        def merge(self, other):
            for i in range(self.k):
                self.sets[i].merge(other.sets[self.k - 1 - i])
            self.sets = sorted(self.sets, reverse=True)

        @property
        def spread(self) -> int:
            return self.sets[0].sum - self.sets[-1].sum

        def __lt__(self, other):
            # least heap, let the state with largest spread to be popped first,
            # if the spread is the same, let the state who has the largest set
            # to be popped first.
            if self.spread != other.spread:
                return self.spread > other.spread
            return self.sets[0] > other.sets[0]

        def __repr__(self) -> str:
            repr_str = "["
            for i in range(self.k):
                if i > 0:
                    repr_str += ","
                repr_str += "{"
                for j, (_, seqlen) in enumerate(self.sets[i].items):
                    if j > 0:
                        repr_str += ","
                    repr_str += str(seqlen)
                repr_str += "}"
            repr_str += "]"
            return repr_str

    sorted_seq_cost_list = sorted([(seqlen, i) for i, seqlen in enumerate(seq_cost_list)])
    states_pq = []
    if equal_size:
        assert len(seq_cost_list) % k_partitions == 0
        for offset in range(0, len(sorted_seq_cost_list), k_partitions):
            items = []
            for i in range(k_partitions):
                seqlen, idx = sorted_seq_cost_list[offset + i]
                items.append((idx, seqlen))
            heapq.heappush(states_pq, State(items=items, k=k_partitions))
    else:
        for seqlen, idx in sorted_seq_cost_list:
            heapq.heappush(states_pq, State(items=[(idx, seqlen)], k=k_partitions))

    while len(states_pq) > 1:
        state0 = heapq.heappop(states_pq)
        state1 = heapq.heappop(states_pq)
        # merge states
        state0.merge(state1)
        heapq.heappush(states_pq, state0)

    final_state = states_pq[0]
    partitions = final_state.get_partitions()
    if equal_size:
        for i, partition in enumerate(partitions):
            assert len(partition) * k_partitions == len(
                seq_cost_list
            ), f"{len(partition)} * {k_partitions} != {len(seq_cost_list)}"
    return partitions


def greedy_partition(seqlen_list: list[int], k_partitions: int, equal_size: bool):
    bias = sum(seqlen_list) + 1 if equal_size else 0
    sorted_seqlen = [(seqlen + bias, i) for i, seqlen in enumerate(seqlen_list)]
    partitions = [[] for _ in range(k_partitions)]
    partition_sums = [0 for _ in range(k_partitions)]
    for seqlen, i in sorted_seqlen:
        min_idx = None
        for j in range(k_partitions):
            if min_idx is None or partition_sums[j] < partition_sums[min_idx]:
                min_idx = j
        partitions[min_idx].append(i)
        partition_sums[min_idx] += seqlen
    if equal_size:
        for i, partition in enumerate(partitions):
            assert len(partition) * k_partitions == len(
                seqlen_list
            ), f"{len(partition)} * {k_partitions} != {len(seqlen_list)}"
    return partitions


eps = 1e-6


def swap_two_partitions(seq_cost_list: list[int], partitions: list[list[int]]):
    def _swap_two_partitions(partition_1: list[int], partition_2: list[int]):
        max_cost = sum(seq_cost_list[_] for _ in partition_1)
        min_cost = sum(seq_cost_list[_] for _ in partition_2)
        assert max_cost >= min_cost
        for i, idx_1 in enumerate(partition_1):
            cost_1 = seq_cost_list[idx_1]
            for j, idx_2 in enumerate(partition_2):
                cost_2 = seq_cost_list[idx_2]
                if cost_1 - cost_2 <= eps:
                    continue
                if (
                    max_cost - cost_1 + cost_2 > min_cost + eps
                    and min_cost - cost_1 + cost_2 + eps < max_cost
                ):
                    partition_1[i] = idx_2
                    partition_2[j] = idx_1
                    return True
        return False

    for i in range(1, len(partitions) - 1):
        for j in range(i + 1, len(partitions) - 1):
            if _swap_two_partitions(partitions[i], partitions[j]):
                return True
    return False


def swap_max_partition(seq_cost_list: list[int], partitions: list[list[int]]):
    max_cost = sum(seq_cost_list[_] for _ in partitions[0])
    min_cost = sum(seq_cost_list[_] for _ in partitions[-1])
    for i, item_idx in enumerate(partitions[0]):
        item_cost = seq_cost_list[item_idx]
        for j in range(1, len(partitions)):
            cost_j = sum(seq_cost_list[_] for _ in partitions[j])
            for k, swap_idx in enumerate(partitions[j]):
                swap_cost = seq_cost_list[swap_idx]
                if item_cost - swap_cost <= eps:
                    continue
                if (
                    cost_j - swap_cost + item_cost + eps < max_cost
                    and max_cost - item_cost + swap_cost > min_cost + eps
                ):
                    partitions[j][k] = item_idx
                    partitions[0][i] = swap_idx
                    return True
    return False


def swap_min_partition(seq_cost_list: list[int], partitions: list[list[int]]):
    max_cost = sum(seq_cost_list[_] for _ in partitions[0])
    min_cost = sum(seq_cost_list[_] for _ in partitions[-1])
    for i, item_idx in enumerate(partitions[-1]):
        item_cost = seq_cost_list[item_idx]
        for j in range(0, len(partitions) - 1):
            cost_j = sum(seq_cost_list[_] for _ in partitions[j])
            for k, swap_idx in enumerate(partitions[j]):
                swap_cost = seq_cost_list[swap_idx]
                if swap_cost - item_cost <= eps:
                    continue
                if (
                    cost_j - swap_cost + item_cost > min_cost + eps
                    and min_cost - item_cost + swap_cost + eps < max_cost
                ):
                    partitions[j][k] = item_idx
                    partitions[-1][i] = swap_idx
                    return True
    return False


def print_partitions(seq_cost_list: list[int], partitions: list[list[int]]):
    print("seq_cost_list: ", seq_cost_list)
    for i, partition in enumerate(partitions):
        print(
            f"partition[{i}]: {partition}, sum of workloads: {sum(seq_cost_list[i] for i in partition)}"
        )
    print("-" * 50)


def balance_partition(seq_cost_list: list[int], partitions: list[list[int]]):
    # print("before balance:")
    # print_partitions(seq_cost_list, partitions)
    while True:
        partitions = sorted(
            partitions,
            key=lambda x: (sum(seq_cost_list[i] for i in x), min(x) if x else 0),
            reverse=True,
        )
        if swap_max_partition(seq_cost_list, partitions):
            continue
        if swap_min_partition(seq_cost_list, partitions):
            continue
        break
    # print("after balance:")
    # print_partitions(seq_cost_list, partitions)
    return partitions


def get_seqlen_balanced_partitions(
    seqlen_list: list[int],
    k_partitions: int,
    equal_size: bool,
    get_seq_costs_func=None,
):
    """
    Calculates partitions of indices from seqlen_list such that the sum of sequence lengths
    in each partition is balanced. Uses the Karmarkar-Karp differencing method.

    This is useful for balancing workload across devices or batches, especially when
    dealing with variable sequence lengths.

    Args:
        seqlen_list (List[int]): A list of sequence lengths for each item.
        k_partitions (int): The desired number of partitions.
        equal_size (bool): If True, ensures that each partition has the same number of items.
                           Requires len(seqlen_list) to be divisible by k_partitions.
                           If False, partitions can have varying numbers of items, focusing
                           only on balancing the sum of sequence lengths.

    Returns:
        List[List[int]]: A list containing k_partitions lists. Each inner list contains the
                         original indices of the items assigned to that partition. The indices
                         within each partition list are sorted.

    Raises:
        AssertionError: If len(seqlen_list) < k_partitions.
        AssertionError: If equal_size is True and len(seqlen_list) is not divisible by k_partitions.
        AssertionError: If any resulting partition is empty.
    """
    if get_seq_costs_func is None:
        get_seq_costs_func = get_seq_costs_linear
    seq_cost_list = get_seq_costs_func(seqlen_list)
    assert (
        len(seq_cost_list) >= k_partitions
    ), f"number of items:[{len(seq_cost_list)}] < k_partitions:[{k_partitions}]"

    def _check_and_sort_partitions(partitions):
        assert len(partitions) == k_partitions, f"{len(partitions)} != {k_partitions}"
        seen_idx = set()
        sorted_partitions = [None] * k_partitions
        for i, partition in enumerate(partitions):
            assert len(partition) > 0, f"the {i}-th partition is empty"
            for idx in partition:
                seen_idx.add(idx)
            sorted_partitions[i] = sorted(partition)
        assert seen_idx == set(range(len(seq_cost_list)))
        return sorted_partitions

    def _get_workload_diff(partitions):
        partition_workloads = [sum(seq_cost_list[i] for i in partition) for partition in partitions]
        return max(partition_workloads) - min(partition_workloads)

    partitions = karmarkar_karp(
        seq_cost_list=seq_cost_list, k_partitions=k_partitions, equal_size=equal_size
    )
    partitions = balance_partition(seq_cost_list, partitions)
    if not equal_size and len(seq_cost_list) % k_partitions == 0:
        equal_partitions = karmarkar_karp(
            seq_cost_list=seq_cost_list, k_partitions=k_partitions, equal_size=True
        )
        equal_partitions = balance_partition(seq_cost_list, equal_partitions)
        if _get_workload_diff(partitions) > _get_workload_diff(equal_partitions):
            partitions = equal_partitions
    return _check_and_sort_partitions(partitions)


def ceildiv(a, b):
    return -(a // -b)


def roundup_divisible(a, b):
    return ((a + b - 1) // b) * b


def rearrange_micro_batches(
    seq_len_effective: List[int],
    max_token_len: int,
    dp_group=None,
    same_num_in_dp=True,
    sort_partition_workload=True,
    get_seq_costs_func=None,
):
    if get_seq_costs_func is None:
        get_seq_costs_func = get_seq_costs_linear
    total_seqlen = sum(seq_len_effective)
    # NOTE: num_microbatches <= batch_size, so take the min of this two.
    num_micro_batches = min(len(seq_len_effective), ceildiv(total_seqlen, max_token_len))
    if dist.is_initialized() and same_num_in_dp:
        num_micro_batches = torch.tensor([num_micro_batches]).cuda()
        dist.all_reduce(num_micro_batches, op=dist.ReduceOp.MAX, group=dp_group)
        num_micro_batches = num_micro_batches.cpu().item()

    # Guarantee that the sum of sequence lengths in each partition is less than max_token_len
    while True:
        assert num_micro_batches <= len(
            seq_len_effective
        ), f"{num_micro_batches} <= {len(seq_len_effective)}"
        micro_bsz_idx = get_seqlen_balanced_partitions(
            seq_len_effective,
            num_micro_batches,
            equal_size=False,
            get_seq_costs_func=get_seq_costs_func,
        )
        check_failed = False
        for partition in micro_bsz_idx:
            cur_sum = sum([seq_len_effective[i] for i in partition])
            if cur_sum > max_token_len:  # should satisfy the memory constraint
                check_failed = True
                break
        actual_size = num_micro_batches + 1 if check_failed else len(micro_bsz_idx)
        if dist.is_initialized() and same_num_in_dp:
            actual_size = torch.tensor([actual_size]).cuda()
            dist.all_reduce(actual_size, op=dist.ReduceOp.MAX, group=dp_group)
            actual_size = actual_size.cpu().item()
        if actual_size == num_micro_batches:
            break
        num_micro_batches += 1

    if sort_partition_workload:
        # Use the sum of squared sequence lengths to approximate attention computation workload
        micro_bsz_idx.sort(
            key=lambda partition: (
                sum(get_seq_costs_func([seq_len_effective[idx] for idx in partition])),
                min(partition) if partition else 0,
            ),
            reverse=True,
        )

    return micro_bsz_idx


def get_seq_costs_linear(seq_len: List[int]):
    costs = [s for s in seq_len]
    return costs


def get_seq_costs_fit(seq_len: List[int]):
    """
    1.5b model:
    a = 1.982122
    b = 2.611821
    c = 0.666759
    7b model:
    a = 4.348357
    b = 8.768377
    c = 0.950719
    """
    normalized_seq_len = [s / 32000 for s in seq_len]
    # costs = [1.982122 * s * s + 2.611821 * s for s in normalized_seq_len]
    costs = [4.348357 * s * s + 8.768377 * s for s in normalized_seq_len]
    return costs


def _get_seq_costs_flops(model, seq_len):
    """Compute flops of the model"""

    def compute_flops(seq):
        linear_flops = 0
        # flops of Linear layers
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                linear_flops += module.weight.numel()
        attn_flops = 0
        for name, module in model.named_modules():
            if hasattr(module, "q_proj"):
                attn_flops += module.q_proj.out_features
        assert attn_flops > 0
        # print(f"Linear flops: {linear_flops}, Attn flops: {attn_flops}")
        # flops = (linear_flops + attn_flops * seq) * seq
        flops = (linear_flops / attn_flops + seq) / 32000 * seq / 32000
        return flops

    return [compute_flops(_) for _ in seq_len]


def get_seq_costs_flops(seq_len):
    raise NotImplementedError("get_seq_costs_flops is not implemented")


def get_packing_function(packing_method, *args, **kwargs):
    cost_model = get_args().cost_model
    if cost_model == "linear":
        get_seq_costs_func = get_seq_costs_linear
    elif cost_model == "flops":
        get_seq_costs_func = get_seq_costs_flops
    elif cost_model == "fit":
        get_seq_costs_func = get_seq_costs_fit
    else:
        raise ValueError(f"Cost model {cost_model} not supported")
    if packing_method in {"DebugUneven"}:
        assert (
            os.environ.get("ODC", "0") == "1"
        ), "Packing method that results in different number of micro batches per minibatch requires ODC to be enabled"
    if packing_method == "None":
        function = none_packing
    elif packing_method == "DynamicSameMicro":
        max_token_len = get_args().max_token_len
        function = partial(
            dynamic_packing,
            max_token_len=max_token_len,
            same_micro_num=True,
            use_bin_packing=False,
            get_seq_costs_func=get_seq_costs_func,
        )
    elif packing_method == "DynamicDiffMicro":
        max_token_len = get_args().max_token_len
        assert (
            os.environ.get("ODC", "0") == "1"
        ), "Packing method that results in different number of micro batches per minibatch requires ODC to be enabled"
        function = partial(
            dynamic_packing,
            max_token_len=max_token_len,
            same_micro_num=False,
            use_bin_packing=False,
            get_seq_costs_func=get_seq_costs_func,
        )
    elif packing_method == "LocalSort":
        function = local_sort_packing
    elif packing_method == "DebugUneven":
        function = debug_even_packing
    else:
        raise ValueError(f"Packing method {packing_method} not supported")

    def wrapper(lengths, rank, world_size):
        return function(lengths, rank, world_size, *args, **kwargs)

    return wrapper


def none_packing(lengths, rank, world_size):
    idx = range(len(lengths))
    return [[idx[i]] for i in idx[rank::world_size]]


def dynamic_packing(
    lengths,
    rank,
    world_size,
    max_token_len,
    same_micro_num=False,
    use_bin_packing=False,
    get_seq_costs_func=None,
):
    assert max(lengths) <= max_token_len, f"{max(lengths)} <= {max_token_len}"
    length_np = np.array(lengths, dtype=np.int32)
    # we enforce equal partition size to guarantee a valid solution exist for the same micro num
    mini_partitions = get_seqlen_balanced_partitions(
        lengths,
        world_size,
        equal_size=same_micro_num,
        get_seq_costs_func=get_seq_costs_func,
    )
    min_workload = int(1e18)
    max_workload = 0
    # if rank == 0:
    #     print(f"rank: {rank}, {lengths}")
    for r_i, partition in enumerate(mini_partitions):
        workloads = get_seq_costs_func([lengths[i] for i in partition])
        workloads = [round(w, 2) for w in workloads]
        if r_i == rank:
            print(f"partition[{r_i}]: {partition}, workloads: {workloads}, sum: {sum(workloads)}")
        workload = sum(workloads)
        min_workload = min(min_workload, workload)
        max_workload = max(max_workload, workload)
    if rank == 0:
        print(
            f"min_workload: {min_workload}, max_workload: {max_workload}, diff: {(max_workload - min_workload) / (min_workload + 1e-10)}"
        )
    local_mini_index_np = np.array(mini_partitions[rank], dtype=np.int32)
    local_mini_length_np = length_np[local_mini_index_np]
    if not same_micro_num:  # for odc
        if use_bin_packing:  # guarantee the least number of micros
            micro_num, micro_indexes, micro_lengths = bin_packing_best_fit_decreasing(
                max_token_len, local_mini_length_np.tolist()
            )
        else:  # balance the workload between micros
            micro_indexes = rearrange_micro_batches(
                local_mini_length_np.tolist(),
                max_token_len,
                same_num_in_dp=False,
                sort_partition_workload=True,
                get_seq_costs_func=get_seq_costs_func,
            )
        local_idx = [local_mini_index_np[micro_idx].tolist() for micro_idx in micro_indexes]
    else:  # for baseline
        micro_indexes = rearrange_micro_batches(
            local_mini_length_np.tolist(),
            max_token_len,
            same_num_in_dp=True,
            sort_partition_workload=True,
            get_seq_costs_func=get_seq_costs_func,
        )
        local_idx = [local_mini_index_np[micro_idx].tolist() for micro_idx in micro_indexes]
    return local_idx


def local_sort_packing(lengths, rank, world_size):
    idx = range(len(lengths))
    local_idx = idx[rank::world_size]
    local_idx = sorted(local_idx, key=lambda x: lengths[x])
    return [[idx] for idx in local_idx]


def debug_even_packing(lengths, rank, world_size):
    idx = range(len(lengths))
    local_idx = idx[rank::world_size]
    if rank % 2 == 0:
        micro_batch_size = 1
    else:
        micro_batch_size = 2
    return [local_idx[i : i + micro_batch_size] for i in range(0, len(local_idx), micro_batch_size)]


if __name__ == "__main__":
    import random

    max_token_len = 24000
    test_case_1 = [
        3935,
        4591,
        5994,
        6805,
        7086,
        8335,
        9503,
        10414,
        11438,
        12668,
        13060,
        14036,
        15997,
        16164,
        17815,
        18436,
        19247,
        20807,
        21465,
        22502,
        23177,
        24181,
        25622,
        26959,
    ]
    test_case_2 = [
        2124,
        3614,
        2676,
        5724,
        4391,
        7025,
        7563,
        9595,
        8410,
        9059,
        12683,
        12109,
        12728,
        15604,
        16704,
        17581,
        17335,
        18072,
        20323,
        19439,
        21038,
        22028,
        22641,
        23787,
    ]
    test_case_3 = [
        1703,
        6758,
        4134,
        8195,
        8167,
        9043,
        10393,
        9567,
        8820,
        14510,
        14469,
        16277,
        16042,
        14128,
        15989,
        17185,
        18315,
        22962,
        19951,
        22917,
        23478,
        23601,
        24000,
        24000,
    ]
    test_case_4 = [
        134,
        4340,
        7768,
        8056,
        7682,
        5980,
        10714,
        9368,
        8042,
        10225,
        10101,
        12527,
        12955,
        14022,
        19783,
        20110,
    ]

    test_cases = []
    for _ in range(10):
        test_cases.append([i * 1000 + random.randint(0, 6000) for i in range(16)])
    test_cases.extend([test_case_1, test_case_2, test_case_3, test_case_4])

    get_seq_costs_func = get_seq_costs_linear

    for test_case in test_cases:
        lengths = [min(l, max_token_len) for l in test_case]
        print(lengths)
        min_workload = int(1e18)
        max_workload = 0
        for i in range(8):
            local_idx = dynamic_packing(
                lengths,
                i,
                8,
                max_token_len=max_token_len,
                same_micro_num=True,
                use_bin_packing=False,
                get_seq_costs_func=get_seq_costs_func,
            )
            local_lengths = [lengths[j] for j in sum(local_idx, [])]
            local_workload = get_seq_costs_func(local_lengths)
            print(local_idx, "\t|\t", local_lengths, "\t|\t", sum(local_workload))
            min_workload = min(min_workload, sum(local_workload))
            max_workload = max(max_workload, sum(local_workload))
        print(
            "[Baseline] min_workload: ",
            min_workload,
            "max_workload: ",
            max_workload,
            "diff: ",
            (max_workload - min_workload) / min_workload,
        )
        min_workload = int(1e18)
        max_workload = 0
        for i in range(8):
            local_idx = dynamic_packing(
                lengths,
                i,
                8,
                max_token_len=max_token_len,
                same_micro_num=False,
                use_bin_packing=False,
                get_seq_costs_func=get_seq_costs_func,
            )
            local_lengths = [lengths[j] for j in sum(local_idx, [])]
            local_workload = get_seq_costs_func(local_lengths)
            print(local_idx, "\t|\t", local_lengths, "\t|\t", sum(local_workload))
            min_workload = min(min_workload, sum(local_workload))
            max_workload = max(max_workload, sum(local_workload))
        print(
            "[ODC] min_workload: ",
            min_workload,
            "max_workload: ",
            max_workload,
            "diff: ",
            (max_workload - min_workload) / min_workload,
        )
        print("\n", "-" * 50, "\n")

    print(32000 * 2.611821 / 1536 / 1.982122)
