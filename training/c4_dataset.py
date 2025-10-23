import os, json, glob, collections, torch
from torch.utils.data import IterableDataset, get_worker_info

class C4Dataset(IterableDataset):
    def __init__(self, path, max_length=8000):
        self.path = path
        self.max_length = max_length

    def _rank_world(self):
        # Try torch.distributed first (works with Accelerate)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_rank(), torch.distributed.get_world_size()
        # Fallback to env vars set by many launchers (incl. accelerate)
        rank = int(os.environ.get("RANK", "0"))
        world = int(os.environ.get("WORLD_SIZE", "1"))
        return rank, world

    def _shard_files(self, files):
        # 1) shard by process rank (GPUs)
        rank, world = self._rank_world()
        files = files[rank::world]

        # 2) shard by DataLoader worker within this process
        wi = get_worker_info()
        if wi is not None and wi.num_workers > 1:
            files = files[wi.id::wi.num_workers]
        return files

    def __iter__(self):
        # Enumerate files once per worker
        file_pattern = os.path.join(self.path, '**', '*.json')
        files = sorted(glob.glob(file_pattern, recursive=True))

        files = self._shard_files(files)

        for fp in files:
            with open(fp, 'r') as f:
                for line in f:
                    data = json.loads(line)
                    text = data['text'][:self.max_length]
                    yield {'input_ids': text}

    def collate_fn(self, batch):
        batched = collections.defaultdict(list)
        for data in batch:
            for k, v in data.items():
                batched[k].append(v)
        return batched
