import os
import json
import glob
import collections
import torch


class C4Dataset(torch.utils.data.IterableDataset):
    def __init__(self, path, max_length=8000):
        self.path = path
        self.max_length = max_length

    def __iter__(self):
        # Get worker info
        worker_info = torch.utils.data.get_worker_info()

        # Find all json files in the directory
        file_pattern = os.path.join(self.path, '**', '*.json')
        files = sorted(glob.glob(file_pattern, recursive=True))

        # Partition files among workers
        if worker_info is not None:
            # Multiple workers: split files among them
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            files = files[worker_id::num_workers]  # Each worker gets every Nth file

        # Iterate through this worker's assigned files
        for file_path in files:
            with open(file_path, 'r') as f:
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
