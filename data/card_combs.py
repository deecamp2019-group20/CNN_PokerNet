# -*- coding: utf-8 -*-
"""
1                        13  15
3 4 5 6 7 8 9 10 J Q K A 2 X D

PASS                1
Rocket              1
Single              15
Double              13
Trio                13
Bomb                13
TrioSingle          13*14 = 182
TrioDouble          13*12 = 156
SingleChain         8+7+6+5+4+3+2+1 = 36
DoubleChain         10+9+8+7+6+5+4+3 = 52
TrioChain(feiji)    11+10+9+8+7 = 45
Quadr2Single        13*(C(14,2)+12) = 1339
Quadr2Double        13*C(12,2) = 858
PlaneSingleWing     11*C(13,2)+10*C(12,3)+9*C(11,4)+8*C(10,5) = 8044
PlaneDoubleWing     11*C(11,2)+10*C(10,3)+9*C(9,4) = 2939

Total               13707
"""

import pandas as pd
from itertools import combinations, combinations_with_replacement
# from collections import defaultdict
from os.path import join, abspath, dirname, exists
from collections import namedtuple
import copy

card = [str(i) for i in range(3, 14)] + ['1', '2', '14', '15']
name_to_rank = {
    '3': 1, '4': 2, '5': 3, '6': 4,
    '7': 5, '8': 6, '9': 7, '10': 8,
    '11': 9, '12': 10, '13': 11,
    '1': 12, '2': 13, '14': 14, '15': 15
}
rank_to_name = {
    v: k for k, v in name_to_rank.items()
}
type_encoding = {
    'buyao': 0, 'dan': 1, 'dui': 2, 'san': 3, 'san_yi': 4,
    'san_er': 5, 'dan_shun': 6, 'er_shun': 7, 'feiji': 8,
    'xfeiji': 9, 'dfeiji': 10, 'zha': 11, 'si_erdan': 12,
    'si_erdui': 13, 'wangzha': 14
}
type_encoding1 = {
    'PASS': 0, 'Rocket': 1, 'Single': 2, 'Double': 3, 'Trio': 4,
    'Bomb': 5, 'TrioSingle': 6, 'TrioDouble': 7, 'SingleChain': 8,
    'DoubleChain': 9, 'TrioChain': 10, 'Quadr2Single': 11,
    'Quadr2Double': 12, 'PlaneSingleWing': 13, 'PlaneDoubleWing': 14
}
inv_type_encoding = {
    v: k for k, v in type_encoding.items()
}

# Card Rank with Card Color
all_card_type = [
    '1-a', '1-b', '1-c', '1-d',
    '2-a', '2-b', '2-c', '2-d',
    '3-a', '3-b', '3-c', '3-d',
    '4-a', '4-b', '4-c', '4-d',
    '5-a', '5-b', '5-c', '5-d',
    '6-a', '6-b', '6-c', '6-d',
    '7-a', '7-b', '7-c', '7-d',
    '8-a', '8-b', '8-c', '8-d',
    '9-a', '9-b', '9-c', '9-d',
    '10-a', '10-b', '10-c', '10-d',
    '11-a', '11-b', '11-c', '11-d',
    '12-a', '12-b', '12-c', '12-d',
    '13-a', '13-b', '13-c', '13-d',
    '14-a', '15-a'
]

# [---name---] main type sum kicker


def dan():
    f = pd.DataFrame(columns=card, dtype=int)
    for k in card:
        f = f.append(
            {k: 1, 'main': name_to_rank[k], 'kicker': "[]"},
            ignore_index=True
        )
    f['type'] = 'dan'
    print(
        'length of generated dan: {}'
        .format(len(f))
    )
    print(
        'Cards: {}'.format(card)
    )
    assert(len(f) == 15)
    return f


def dui():
    f = pd.DataFrame(columns=card, dtype=int)
    for k in card[:-2]:
        f = f.append(
            {k: 2, 'main': name_to_rank[k], 'kicker': "[]"},
            ignore_index=True
        )
    f['type'] = 'dui'
    print(
        'length of generated dui: {}'
        .format(len(f))
    )
    print(
        'Cards: {}'.format(card)
    )
    assert(len(f) == 13)
    return f


def san():
    f = pd.DataFrame(columns=card, dtype=int)
    for k in card[:-2]:
        f = f.append(
            {k: 3, 'main': name_to_rank[k], 'kicker': "[]"},
            ignore_index=True)
    f['type'] = 'san'
    print(
        'length of generated san: {}'
        .format(len(f))
    )
    print(
        'Cards: {}'.format(card)
    )
    assert(len(f) == 13)
    return f


# NOTE: Modified combinations list(set)
def san_yi():
    f = pd.DataFrame(columns=card, dtype=int)
    for k in card[:-2]:
        buffer_card = copy.deepcopy(card)
        buffer_card.remove(k)
        # comb = combinations(list(set(card)-{k}), 1)
        comb = combinations(buffer_card, 1)
        for c in comb:
            f = f.append(
                {k: 3, c[0]: 1, 'main': name_to_rank[k], 'kicker': [c[0]]},
                ignore_index=True
            )
    f['type'] = 'san_yi'
    print(
        'length of generated san_yi: {}'
        .format(len(f))
    )
    print(
        'Cards: {}'.format(card)
    )
    assert(len(f) == 182)
    return f


# NOTE: Modified combinations list(set)
def san_er():
    f = pd.DataFrame(columns=card, dtype=int)
    for k in card[:-2]:
        # buffer_card = card[:-2]
        buffer_card = copy.deepcopy(card[:-2])
        buffer_card.remove(k)
        # comb = combinations(list(set(card[:-2])-{k}), 1)
        comb = combinations(buffer_card, 1)
        for c in comb:
            if c[0] != k:
                f = f.append(
                    {k: 3, c[0]: 2, 'main': name_to_rank[k], 'kicker': [c[0]]},
                    ignore_index=True)
    f['type'] = 'san_er'
    print(
        'length of generated san_er: {}'
        .format(len(f))
    )
    print('Cards: {}'.format(card))
    assert(len(f) == 156)
    return f


def dan_shun():
    f = pd.DataFrame(columns=card, dtype=int)
    for L in range(5, 13):
        for i in range(5+8-L):
            sli = card[i: i+L]
            data = {k: 1 for k in sli}
            data['main'] = name_to_rank[sli[0]]
            data['kicker'] = [sli[-1]]
            f = f.append(data, ignore_index=True)
    f['type'] = 'dan_shun'
    print(
        'length of generated dan_shun: {}'
        .format(len(f))
    )
    print(
        'Cards: {}'.format(card)
    )
    assert(len(f) == 36)
    return f


def er_shun():
    f = pd.DataFrame(columns=card, dtype=int)
    for L in range(3, 11):
        for i in range(3+10-L):
            sli = card[i: i+L]
            data = {k: 2 for k in sli}
            data['main'] = name_to_rank[sli[0]]
            data['kicker'] = [sli[-1]]
            f = f.append(data, ignore_index=True)
    f['type'] = 'er_shun'
    print(
        'length of generated er_shun: {}'
        .format(len(f))
    )
    print(
        'Cards: {}'.format(card)
    )
    assert(len(f) == 52)
    return f


# Which could also be treat as TrioChain
def feiji():
    f = pd.DataFrame(columns=card, dtype=int)
    for L in range(2, 7):
        for i in range(2+11-L):
            sli = card[i: i+L]
            data = {k: 3 for k in sli}
            data['main'] = name_to_rank[sli[0]]
            data['kicker'] = [sli[-1]]
            # data['kicker'] = list(sli[-1])
            f = f.append(data, ignore_index=True)
    f['type'] = 'feiji'
    print(
        'length of generated feiji: {}'
        .format(len(f))
    )
    print(
        'Cards: {}'.format(card)
    )
    assert(len(f) == 45)
    return f


# NOTE: Modified combinations list(set)
def xfeiji():
    f = pd.DataFrame(columns=card, dtype=int)
    for L in range(2, 6):
        for i in range(2+11-L):
            sli = card[i: i+L]
            # buffer_card = card
            buffer_card = copy.deepcopy(card)
            for s in sli:
                buffer_card.remove(s)
            # comb = combinations(list(set(card)-set(sli)), L)
            comb = combinations(buffer_card, L)
            for c in comb:
                data = {k: 1 for k in c}
                data.update({k: 3 for k in sli})
                data['main'] = name_to_rank[sli[0]]
                data['kicker'] = list(c)
                f = f.append(data, ignore_index=True)
    f['type'] = 'xfeiji'
    print(
        'length of generated xfeiji: {}'
        .format(len(f))
    )
    print(
        'Cards: {}'.format(card)
    )
    assert(len(f) == 8044)
    return f


# NOTE: Modified combinations list(set)
def dfeiji():
    f = pd.DataFrame(columns=card, dtype=int)
    for L in range(2, 5):
        for i in range(2+11-L):
            sli = card[i: i+L]
            # buffer_card = card[:-2]
            buffer_card = copy.deepcopy(card[:-2])
            for s in sli:
                buffer_card.remove(s)
            # comb = combinations(list(set(card[:-2])-set(sli)), L)
            comb = combinations(buffer_card, L)
            for c in comb:
                data = {k: 2 for k in c}
                data.update({k: 3 for k in sli})
                data['main'] = name_to_rank[sli[0]]
                data['kicker'] = list(c)
                f = f.append(data, ignore_index=True)
    f['type'] = 'dfeiji'
    print(
        'length of generated dfeiji: {}'
        .format(len(f))
    )
    print(
        'Cards: {}'.format(card)
    )
    assert(len(f) == 2939)
    return f


def zha():
    f = pd.DataFrame(columns=card, dtype=int)
    for k in card[:-2]:
        f = f.append(
            {k: 4, 'main': name_to_rank[k], 'kicker': []},
            ignore_index=True)
    f['type'] = 'zha'
    print(
        'length of generated zha: {}'
        .format(len(f))
    )
    print(
        'Cards: {}'.format(card)
    )
    assert(len(f) == 13)
    return f


# NOTE
# 1. Modified to add same single cards
# 2. Modified combinations list(set)
def si_erdan():
    f = pd.DataFrame(columns=card, dtype=int)
    for k in card[:-2]:
        # buffer_card = card
        buffer_card = copy.deepcopy(card)
        buffer_card.remove(k)
        # comb = combinations(list(set(card)-{k}), 2)
        # comb = combinations(buffer_card, 2)
        comb = combinations_with_replacement(buffer_card, 2)
        for c in comb:
            if c[0] == c[1] and c[0] != '14' and c[0] != '15':
                f = f.append({
                    k: 4, c[0]: 2,
                    'main': name_to_rank[k], 'kicker': list(c)},
                    ignore_index=True
                )
            elif c[0] == c[1] and (c[0] == '14' or c[0] == '15'):
                pass
            else:
                f = f.append({
                    k: 4, c[0]: 1, c[1]: 1,
                    'main': name_to_rank[k], 'kicker': list(c)},
                    ignore_index=True
                )
                # f = f.append({
                #     k: 4, c[0]: 1, c[1]: 1,
                #     'main': name_to_rank[k], 'kicker': list(c)},
                #     ignore_index=True)
    f['type'] = 'si_erdan'
    print(
        'length of the generated si_erdan: {}'
        .format(len(f))
    )
    print(
        'Cards: {}'.format(card)
    )
    # assert(len(f) == 1183)
    assert(len(f) == 1339)
    return f


# NOTE: Modified combinations list(set)
def si_erdui():
    f = pd.DataFrame(columns=card, dtype=int)
    for k in card[:-2]:
        # buffer_card = card[:-2]
        buffer_card = copy.deepcopy(card[:-2])
        buffer_card.remove(k)
        # comb = combinations(list(set(card[:-2])-{k}), 2)
        comb = combinations(buffer_card, 2)
        for c in comb:
            f = f.append({
                k: 4, c[0]: 2, c[1]: 2,
                'main': name_to_rank[k], 'kicker': list(c)},
                ignore_index=True)
    f['type'] = 'si_erdui'
    print(
        'length of the generated si_erdui: {}'.
        format(len(f))
    )
    print(
        'Cards: {}'.format(card)
    )
    assert(len(f) == 858)
    return f


def wangzha():
    f = pd.DataFrame(columns=card, dtype=int)
    f = f.append({
        '14': 1, '15': 1, 'main': 15, 'kicker': []},
        ignore_index=True)
    f['type'] = 'wangzha'
    assert(len(f) == 1)
    return f


def buyao():
    f = pd.DataFrame(columns=card, dtype=int)
    data = {k: 0 for k in card}
    data.update({'main': 0, 'type': 'buyao', 'kicker': []})
    f = f.append(data, ignore_index=True)
    assert(len(f) == 1)
    return f


def calc_key(row):
    s = []
    for k in card:
        s.extend([int(k)] * row[k])
    s = str(sorted(s))
    return s


if __name__ == "__main__":
    if exists(join(dirname(abspath(__file__)), "patterns.csv")):
        All = pd.read_csv(
            join(dirname(abspath(__file__)), "patterns.csv")).fillna(0)
    else:
        # tmp = pd.concat([
        #     dan(), dui(), san(), san_yi(), san_er(),
        #     dan_shun(), er_shun(), feiji(), xfeiji(), dfeiji(),
        #     zha(), si_erdan(), si_erdui(), wangzha(), buyao()],
        #     axis=0, sort=False).fillna(0)
        tmp = pd.concat([
            buyao(), wangzha(), dan(), dui(), san(),
            zha(), san_yi(), san_er(), dan_shun(), er_shun(),
            feiji(), si_erdan(), si_erdui(), xfeiji(), dfeiji()],
            axis=0, sort=False).fillna(0)

        tmp['sum'] = tmp[card].sum(axis=1)

        All = tmp.drop(['type', 'kicker'], axis=1).astype(int)
        All['type'] = tmp['type']
        All['kicker'] = tmp['kicker']
        All['key'] = All.apply(calc_key, axis=1)

        All.to_csv(
            join(dirname(abspath(__file__)), "patterns.csv"),
            index=False)

    move_desc = namedtuple('move_desc', ['type', 'sum', 'main', 'kicker'])
    cache = {}
    for row in All.itertuples():
        cache[row.key] = move_desc(
            type=row.type, sum=row.sum, main=row.main, kicker=eval(row.kicker)
        )
