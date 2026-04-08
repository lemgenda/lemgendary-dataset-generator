import random
from collections import defaultdict

class SmartSampler:
    def __init__(self, dataset, balance=True):
        self.dataset = dataset
        self.balance = balance
        self.task_buckets = defaultdict(list)
        for i, item in enumerate(dataset.metadata):
            self.task_buckets[item["task"]].append(i)
        
        self.tasks = list(self.task_buckets.keys())

    def sample(self):
        if self.balance:
            task = random.choice(self.tasks)
            idx = random.choice(self.task_buckets[task])
        else:
            idx = random.randint(0, len(self.dataset) - 1)
        
        return idx