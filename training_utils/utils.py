from collections import deque, defaultdict
from numbers import Number
from typing import Iterable, List
from functools import partial

class IntstanceAvgMeter:
    def __init__(self, window_size=100):
        self.window_size = window_size
        self.results = defaultdict(lambda : deque(maxlen=window_size))

    def _mean(self, values: List[Number]) -> float:
        return sum(values) / len(values) if values else 0
    
    def update(self, key: str, value: Number):
        self.results[key].append(value)

    def instance_avg(self, key: str) -> float:
        return self._mean(self.results[key])

    def all_avg(self) -> float:
        return self._mean([self.instance_avg(key) for key in self.results.keys() if self.results[key]])
    
    def all_max_avg(self) -> float:
        return self._mean([max(val) for val in self.results.values() if val])

    def all_min_avg(self) -> float:
        return self._mean([max(val) for val in self.results.values() if val])

    def reset(self):
        self.results = defaultdict(lambda : deque(maxlen=self.window_size))
