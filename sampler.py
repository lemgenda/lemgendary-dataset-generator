import random
import os
from pathlib import Path
from collections import defaultdict

class SmartSampler:
    """
    SOTA 2026 Smart Sampler: Multi-dimensional balancing engine.
    Balances across Classes, Source Datasets, and Model Tasks.
    """
    def __init__(self, dataset, balance_type="multi"):
        self.dataset = dataset
        self.balance_type = balance_type # "multi", "task", "class", or "none"
        self.metadata = dataset.metadata
        
        # 1. Bucketization
        self.task_buckets = defaultdict(list)
        self.source_buckets = defaultdict(list)
        self.class_buckets = defaultdict(list)
        
        print(f"📡 [SAMPLER] Indexing {len(self.metadata)} items for balanced velocity...")
        for i, item in enumerate(self.metadata):
            # Task Balancing
            self.task_buckets[item["task"]].append(i)
            # Source (Dataset) Balancing
            self.source_buckets[item["source"]].append(i)
            
            # Class Balancing (if detection labels exist)
            label_path = item.get("label_path")
            if label_path and os.path.exists(label_path):
                with open(label_path) as f:
                    for line in f:
                        if line.strip():
                            cls = line.split()[0]
                            self.class_buckets[cls].append(i)

        self.tasks = list(self.task_buckets.keys())
        self.sources = list(self.source_buckets.keys())
        self.classes = list(self.class_buckets.keys())

    def sample(self):
        if self.balance_type == "none":
            return random.randint(0, len(self.dataset) - 1)
        
        # 2. Strategic Selection
        # If multi-balancing, we rotate priorities to ensure geometric coverage
        strategy = random.choice(["task", "source", "class"]) if self.classes else random.choice(["task", "source"])
        
        if strategy == "task":
            key = random.choice(self.tasks)
            return random.choice(self.task_buckets[key])
        elif strategy == "source":
            key = random.choice(self.sources)
            return random.choice(self.source_buckets[key])
        elif strategy == "class" and self.classes:
            key = random.choice(self.classes)
            return random.choice(self.class_buckets[key])
            
        return random.randint(0, len(self.dataset) - 1)