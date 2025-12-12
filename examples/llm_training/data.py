import torch
from args import get_args
from datasets import Dataset


class BatchedDataset:
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        rank: int,
        world_size: int,
        packing_method: str = "None",
    ):
        self.dataset = dataset
        self.rank = rank
        self.world_size = world_size
        self.batch_size = batch_size
        self.global_batch_size = batch_size * world_size
        from packing import get_packing_function

        self.packing_function = get_packing_function(packing_method)
        self.limit_dataset_token_len = get_args().limit_dataset_token_len

    def __len__(self):
        return len(self.dataset) // self.global_batch_size

    def __getitem__(self, index):
        indices = range(index * self.global_batch_size, (index + 1) * self.global_batch_size)
        data = self.dataset[indices]
        lengths = [len(item) for item in data["input_ids"]]

        if self.limit_dataset_token_len is not None:
            print(f"Limiting dataset to {self.limit_dataset_token_len} tokens")
            # Filter out sequences longer than limit_dataset_token_len
            valid_indices = [
                i for i, length in enumerate(lengths) if length <= self.limit_dataset_token_len
            ]
            if len(valid_indices) == 0:
                # If all sequences are too long, truncate them instead
                valid_indices = list(range(len(lengths)))
                for i in valid_indices:
                    for key in data.keys():
                        if (
                            isinstance(data[key][i], list)
                            and len(data[key][i]) > self.limit_dataset_token_len
                        ):
                            data[key][i] = data[key][i][: self.limit_dataset_token_len]
                lengths = [
                    min(len(item), self.limit_dataset_token_len) for item in data["input_ids"]
                ]
            else:
                # Use only valid sequences
                data = {key: [data[key][i] for i in valid_indices] for key in data.keys()}
                lengths = [lengths[i] for i in valid_indices]

            # Pad sequences to make count divisible by world_size (required for DynamicSameMicro)
            # This ensures the packing function can partition sequences evenly across ranks
            num_valid = len(lengths)
            remainder = num_valid % self.world_size
            if remainder != 0:
                pad_count = self.world_size - remainder
                # Pad by repeating sequences from the beginning (copy to avoid shared references)
                for i in range(pad_count):
                    pad_idx = i % num_valid
                    for key in data.keys():
                        # Copy the list to avoid shared references
                        if isinstance(data[key][pad_idx], list):
                            data[key].append(data[key][pad_idx].copy())
                        else:
                            data[key].append(data[key][pad_idx])
                    lengths.append(lengths[pad_idx])

        packed_indices = self.packing_function(lengths, self.rank, self.world_size)

        debug_str = f"rank {self.rank} "
        for mb in packed_indices:
            debug_str += (
                "+".join([str(lengths[i]) for i in mb])
                + "="
                + str(sum([lengths[i] for i in mb]))
                + "; "
            )
        all_debug_str = [None for _ in range(self.world_size)]
        torch.distributed.all_gather_object(all_debug_str, debug_str)
        if self.rank == 0:
            print("\n".join(all_debug_str))

        res = []

        for batch_indices in packed_indices:

            batch = [
                {key: torch.tensor(data[key][i]) for key in data.keys()} for i in batch_indices
            ]

            batch = do_packing(batch, self.batch_size)
            res.append(batch)

        return res


def do_packing(batch, minibatch_size):
    seq_lens = [item["attention_mask"].sum() for item in batch]
    input_ids = torch.cat(
        [item["input_ids"][: seq_lens[i]] for i, item in enumerate(batch)]
    ).unsqueeze(0)
    position_ids = torch.cat([torch.arange(seq_lens[i]) for i, item in enumerate(batch)]).unsqueeze(
        0
    )
    attention_mask = torch.cat(
        [item["attention_mask"][: seq_lens[i]] for i, item in enumerate(batch)]
    ).unsqueeze(0)

    loss_scale = []
    for seq_len in seq_lens:
        loss_scale.append(torch.ones(seq_len) / seq_len / minibatch_size)
    loss_scale = torch.cat(loss_scale).unsqueeze(0)
    eos_pos = torch.cumsum(torch.tensor(seq_lens), dim=0) - 1
    loss_scale[:, eos_pos] = 0
    return {
        "input_ids": input_ids,
        "position_ids": position_ids,
        "attention_mask": attention_mask,
        "loss_scale": loss_scale,
        "num_seqs": len(seq_lens),
    }


def main():
    from datasets import load_from_disk

    dataset = load_from_disk("fsdp/data/longalign256")
    batched_dataset = BatchedDataset(
        dataset, batch_size=4, rank=0, world_size=2, packing_method="Fixed"
    )
    print(len(batched_dataset))
    print(batched_dataset[0])


if __name__ == "__main__":
    main()
