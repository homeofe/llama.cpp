#!/usr/bin/env python3
"""Overlapping GPU work and CPU windows must not be counted twice."""
from analyze import union, overlap

busy = union([(0, 10), (5, 15), (20, 30), (30, 40), (99, 99)])
assert busy == [[0, 15], [20, 40]]
assert overlap(busy, 8, 25) == 12
assert 25-8-overlap(busy, 8, 25) == 5
assert overlap(busy, 15, 20) == 0
assert overlap(busy, 2, 5) == 3
assert overlap([], 1, 9) == 0
print('interval accounting: six checks passed')
