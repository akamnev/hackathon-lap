import enum


class BondType(enum.IntEnum):
    SINGLE = 1
    DOUBLE = 2
    TRIPLE = 3
    AROMATIC = 4


IDX2ATOM = [
    1, 3, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 19, 20, 30, 33, 34, 35,
    38, 47, 53
]

ATOM2IDX = {a: i+1 for i, a in enumerate(IDX2ATOM)}
