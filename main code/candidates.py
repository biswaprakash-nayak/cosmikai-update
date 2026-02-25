# This files makes the candidates class so we can store the candidates easily

# last updated: 23-feb-2026
# updated by: Biswaprakash Nayak
# changes made: creating the file and making the class

# heard its better to use dataclasses than normal classes, tho idk if it really is better
from dataclasses import dataclass

# makes each candidate a dataclass for better organization and readability
@dataclass
class Candidate:
    period: float
    t0: float
    duration: float
    depth: float
    power: float