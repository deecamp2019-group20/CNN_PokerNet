import numpy as np
import argparse
import os
# import math
# import lmdb

parser = argparse.ArgumentParser()
parser.add_argument(
    '-i',
    '--inputFile',
    type=str,
    default='./landlord_test.log',
    help='the path towards the log file in order to read data'
)
parser.add_argument(
    '-pid',
    '--personID',
    type=int,
    default=0,
    help=(
        'the ID for the player (winner).'
        '0:landlord, 1:landlor_down, 2:landlord_up')
)
parser.add_argument(
    '-s',
    '--save_dir',
    type=str,
    help='The generated mdb file save dir',
    default='./train'
)
parser.add_argument(
    '-nT',
    '--train_num',
    type=int,
    help='The number of game process to be train dataset',
    default=330000
)
opt = parser.parse_args()
print(opt)


def split_handcards(cards):
    # Split Cards Series into a prettier list
    r""" Handcards string spliter
    Split Cards Series into a prettier list which sorted DESCEDNING

    Args:
        cards: a string, which indicate a group of cards

    Output:
        hand_cards: a string

    """
    hand_cards = []
    cards_rank = [
        '3', '4', '5', '6', '7', '8', '9', '10',
        'J', 'Q', 'K', 'A', '2', 'X', 'D'
    ]
    for card in cards:
        # NOTE: '10' contrains 2 chars which should be seperately considered
        if card != '1' and card != '0':
            hand_cards.append(card)
        elif card == '1':
            hand_cards.append('10')
        elif card == '0':
            pass
        else:
            pass
    # sort
    length = len(hand_cards)
    for index in range(length):
        for i in range(1, length - index):
            if (
                cards_rank.index(hand_cards[i - 1]) <
                    cards_rank.index(hand_cards[i])):
                hand_cards[i-1], hand_cards[i] = hand_cards[i], hand_cards[i-1]

    return hand_cards


def cards_rank_encode(cards):
    r""" Cards rank number encoder
    Convert a card rank list into binary numpy array

    Args:
        cards: A list of card ranks

    Output:
        A numpy array which only contain 0 or 1
        and the size of this array is 15 * 4
    """
    # NOTE. Here we are using the bool type of numpy array
    binary_array = np.zeros((15, 4), dtype=bool)
    card_ranks = [
        '3', '4', '5', '6', '7', '8', '9',
        '10', 'J', 'Q', 'K', 'A', '2', 'X', 'D']

    for card in cards:
        if card != 'P':
            index = card_ranks.index(card)
            for i in range(0, 4):
                if binary_array[index][i]:
                    pass
                else:
                    binary_array[index][i] = 1
                    break

    return binary_array


def have_trio_in_handcard(handcard):
    r""" Find whether there is trio among the handcard

    Args:
        handcard: a list splited handcard numbers

    Return:
        Boolen
    """
    card_ranks = [
        '3', '4', '5', '6', '7', '8', '9', '10',
        'J', 'Q', 'K', 'A', '2'
    ]
    for rank in card_ranks:
        if handcard.count(rank) >= 3:
            return True
    return False


def have_bomb_in_handcard(handcard):
    r""" Find whether there is bomb or rocket among the handcard

    Args:
        handcard: a list splited handcard numbers

    Return:
        Boolen
    """
    card_ranks = [
        '3', '4', '5', '6', '7', '8', '9', '10',
        'J', 'Q', 'K', 'A', '2', 'X', 'D'
    ]
    for rank in card_ranks:
        if handcard.count(rank) == 4:
            return True

    if 'X' in handcard and 'D' in handcard:
        return True

    return False


def have_plane_in_handcard(handcard):
    r""" Find whether there is plane among the handcard

    Args:
        handcard: a list splited handcard numbers

    Return:
        Boolen
    """
    card_ranks = [
        '3', '4', '5', '6', '7', '8', '9', '10',
        'J', 'Q', 'K', 'A', '2'
    ]
    for i in range(0, 11):
        if (
            handcard.count(card_ranks[i]) >= 3 and
                handcard.count(card_ranks[i + 1]) >= 3):
            # NOTE: AAA222 is not a MainGroup for plane
            return True

    return False


def game_process_with_pass(game_process):
    r""" Add Pass into Game Process

    Args:
        game_process: the initial game processing

    Return:
        process_pass: add pass flag into the game processing part
    """
    game_process_list = game_process.split(';')
    game_process_landlord = []
    game_process_landlord_down = []
    game_process_landlord_up = []

    cur_player = '0'

    for game_step in game_process_list:
        game_step_player = game_step.split(',')[0]
        game_step_cards = game_step.split(',')[1]

        if cur_player == game_step_player:
            if cur_player == '0':
                game_process_landlord.append(game_step_cards)
                # print(
                #     'outside while, cur_player: 0'
                #     'game_step_card: {}'.format(game_step_cards)
                # )
                cur_player = '1'
            elif cur_player == '1':
                game_process_landlord_down.append(game_step_cards)
                # print(
                #     'outside while, cur_player: 1'
                #     'game_step_card: {}'.format(game_step_cards)
                # )
                cur_player = '2'
            elif cur_player == '2':
                game_process_landlord_up.append(game_step_cards)
                # print(
                #     'outside while, cur_player: 2'
                #     'game_step_card: {}'.format(game_step_cards)
                # )
                cur_player = '0'
            else:
                raise ValueError(
                    'player could only be 0, 1, 2, got {}'
                    .format(cur_player)
                )
        else:
            while True:
                # find players who passed until the next one played cards
                if cur_player != game_step_player:
                    if cur_player == '0':
                        game_process_landlord.append('P')
                        # print(
                        #     'inside while, cur_player: 0'
                        #     'game_step_player: {}, game_step_cards: {}'
                        #     .format(game_step_player, game_step_cards)
                        # )
                    elif cur_player == '1':
                        game_process_landlord_down.append('P')
                        # print(
                        #     'inside while, cur_player: 1'
                        #     'game_step_player: {}, game_step_cards: {}'
                        #     .format(game_step_player, game_step_cards)
                        # )
                    elif cur_player == '2':
                        game_process_landlord_up.append('P')
                        # print(
                        #     'inside while, cur_player: 2'
                        #     'game_step_player: {}, game_step_cards: {}'
                        #     .format(game_step_player, game_step_cards)
                        # )
                    else:
                        raise ValueError(
                            'player could only be 0, 1, 2, got {}'
                            .format(cur_player)
                        )
                    # move to check next player
                    if cur_player == '0':
                        cur_player = '1'
                        # print('cur_player mismatch. Move from 0 to 1')
                    elif cur_player == '1':
                        cur_player = '2'
                        # print('cur_player mismatch. Move from 1 to 2')
                    elif cur_player == '2':
                        cur_player = '0'
                        # print('cur_player mismatch. Move from 2 to 0')
                else:
                    if cur_player == '0':
                        game_process_landlord.append(game_step_cards)
                        cur_player = '1'
                    elif cur_player == '1':
                        game_process_landlord_down.append(game_step_cards)
                        cur_player = '2'
                    elif cur_player == '2':
                        game_process_landlord_up.append(game_step_cards)
                        cur_player = '0'
                    break

    return (
        game_process_landlord, game_process_landlord_down,
        game_process_landlord_up)


def is_singleChain(card_list):
    r""" Find whether there is a singlechain in list of cards

    Args:
        card_list: a list of card

    Return:
        index: the index of the detected chain among singleChain (start from 0)
        Boolen: whether this card_list is a singleChain
    """
    cards_rank_chain = [
        '3', '4', '5', '6', '7', '8', '9', '10',
        'J', 'Q', 'K', 'A'
    ]
    if len(card_list) < 5:
        return -1, False
    else:
        for i in range(0, len(card_list) - 1):
            if (
                card_list[i] in cards_rank_chain and
                card_list[i + 1] in cards_rank_chain and
                cards_rank_chain.index(card_list[i]) - 1 ==
                    cards_rank_chain.index(card_list[i + 1])):
                pass
            else:
                return -1, False
        chain_start = card_list[-1]
        chain_len = len(card_list)
        if chain_len == 5:
            index = cards_rank_chain.index(chain_start)
        elif chain_len == 6:
            index = cards_rank_chain.index(chain_start) + 8
        elif chain_len == 7:
            index = cards_rank_chain.index(chain_start) + 15
        elif chain_len == 8:
            index = cards_rank_chain.index(chain_start) + 21
        elif chain_len == 9:
            index = cards_rank_chain.index(chain_start) + 26
        elif chain_len == 10:
            index = cards_rank_chain.index(chain_start) + 30
        elif chain_len == 11:
            index = cards_rank_chain.index(chain_start) + 33
        elif chain_len == 12:
            index = cards_rank_chain.index(chain_start) + 35
        else:
            raise ValueError(
                'the simple chain could not reach length beyond 12, got {}'
                .format(card_list)
            )
        return index, True


def is_doubleChain(card_list):
    r""" Find whether there is a double-chain in list of cards

    Args:
        card_list: a list of card

    Return:
        index: the index of the detected chain among doubleChain (start from 0)
        Boolen: whether this card_list is a doubleChain
    """
    cards_rank_chain = [
        '3', '4', '5', '6', '7', '8', '9', '10',
        'J', 'Q', 'K', 'A'
    ]
    if len(card_list) < 6 or len(card_list) % 2 != 0:
        return -1, False
    else:
        for i in range(0, int(len(card_list) / 2) - 1):
            if (
                card_list[2 * i] in cards_rank_chain and
                card_list[2 * (i + 1)] in cards_rank_chain and
                card_list[2 * i] == card_list[2 * i + 1] and
                card_list[2 * (i + 1)] == card_list[2 * (i + 1) + 1] and
                cards_rank_chain.index(card_list[2 * i]) - 1 ==
                    cards_rank_chain.index(card_list[2 * (i + 1)])):
                pass
            else:
                return -1, False

        chain_start = card_list[-1]
        chain_len = int(len(card_list) / 2)
        if chain_len == 3:
            index = cards_rank_chain.index(chain_start)
        elif chain_len == 4:
            index = cards_rank_chain.index(chain_start) + 10
        elif chain_len == 5:
            index = cards_rank_chain.index(chain_start) + 19
        elif chain_len == 6:
            index = cards_rank_chain.index(chain_start) + 27
        elif chain_len == 7:
            index = cards_rank_chain.index(chain_start) + 34
        elif chain_len == 8:
            index = cards_rank_chain.index(chain_start) + 40
        elif chain_len == 9:
            index = cards_rank_chain.index(chain_start) + 45
        elif chain_len == 10:
            index = cards_rank_chain.index(chain_start) + 49
        else:
            raise ValueError(
                'the double chain could not reach len beyond 2*10, got {}'
                .format(card_list)
            )
        return index, True


def is_trioChain(card_list):
    # index start from 0
    r""" Find whether this is a trio-chain in list of cards

    Args:
        card_list: a list of card

    Return:
        index: the index of the detected chain among trioChain (start from 0)
        Boolen: whether this card_list is a trioChain
    """
    cards_rank_chain = [
        '3', '4', '5', '6', '7', '8', '9', '10',
        'J', 'Q', 'K', 'A'
    ]
    if len(card_list) < 6 or len(card_list) % 3 != 0:
        return -1, False
    else:
        for i in range(0, int(len(card_list) / 3) - 1):
            if (
                card_list[3 * i] in cards_rank_chain and
                card_list[3 * (i + 1)] in cards_rank_chain and
                card_list[3 * i] == card_list[3 * i + 1] ==
                card_list[3 * i + 2] and
                card_list[3 * (i + 1)] == card_list[3 * (i + 1) + 1] ==
                card_list[3 * (i + 1) + 2] and
                cards_rank_chain.index(card_list[3 * i]) - 1 ==
                    cards_rank_chain.index(card_list[3 * (i + 1)])):
                pass
            else:
                return -1, False
        chain_start = card_list[-1]
        chain_len = int(len(card_list) / 3)
        if chain_len == 2:
            index = cards_rank_chain.index(chain_start)
        elif chain_len == 3:
            index = cards_rank_chain.index(chain_start) + 11
        elif chain_len == 4:
            index = cards_rank_chain.index(chain_start) + 21
        elif chain_len == 5:
            index = cards_rank_chain.index(chain_start) + 30
        elif chain_len == 6:
            index = cards_rank_chain.index(chain_start) + 38
        else:
            raise ValueError(
                'the trio chain could not reach length beyond 3*6, got {}'
                .format_map(card_list)
            )
        return index, True


def is_quadr2single(card_list):
    # index starts from 1
    # NOTE: the 2 single here could also be one pair of Double (2 SAME single)
    r""" Find whether this is a 4 cards main group with 2 singles in list of cards

    Args:
        card_list: a list of card

    Return:
        index: the index of the detected comb among quadr2single (start from 1)
        Boolen: whether this card_list is a quadr2single
    """
    cards_rank_simple = [
        '3', '4', '5', '6', '7', '8', '9', '10',
        'J', 'Q', 'K', 'A', '2'
    ]
    cards_rank_all = [
        '3', '4', '5', '6', '7', '8', '9', '10',
        'J', 'Q', 'K', 'A', '2', 'X', 'D'
    ]

    if len(card_list) != 6:
        return -1, False
    else:
        if (
            card_list[0] == card_list[1] == card_list[2] == card_list[3] and
            card_list[4] != card_list[0] and
                card_list[5] != card_list[0]):
            main_group_num = card_list[0]
            # kicker_num_1 <= kicker_num_2
            kicker_num_1 = card_list[5]
            kicker_num_2 = card_list[4]
            # calculate index
            index = cards_rank_simple.index(main_group_num) * (91 + 12)
            for i in range(0, cards_rank_simple.index(kicker_num_1)):
                index += (14 - i)
            index += (
                cards_rank_simple.index(kicker_num_2) -
                cards_rank_simple.index(kicker_num_1) + 1
            )
            return index, True

        elif (
            card_list[1] == card_list[2] == card_list[3] == card_list[4] and
            card_list[0] != card_list[1] and
                card_list[5] != card_list[1]):
            main_group_num = card_list[1]
            kicker_num_1 = card_list[5]
            kicker_num_2 = card_list[0]
            # calculate index
            index = cards_rank_simple.index(main_group_num) * (91 + 12)
            for i in range(0, cards_rank_simple.index(kicker_num_1)):
                index += (14 - i)
            index += (
                cards_rank_all.index(kicker_num_2) -
                cards_rank_all.index(kicker_num_1)
            )
            return index, True

        elif (
            card_list[2] == card_list[3] == card_list[4] == card_list[5] and
            card_list[0] != card_list[2] and
                card_list[1] != card_list[2]):
            main_group_num = card_list[2]
            kicker_num_1 = card_list[1]
            kicker_num_2 = card_list[0]
            # calculate index
            index = cards_rank_simple.index(main_group_num) * (91 + 12)
            for i in range(0, cards_rank_all.index(kicker_num_1)):
                if i < cards_rank_simple.index(main_group_num):
                    index += (14 - i)
                elif i == cards_rank_simple.index(main_group_num):
                    pass
                else:
                    index += (15 - i)
            if kicker_num_1 == 'X':
                index += (
                    cards_rank_all.index(kicker_num_2) -
                    cards_rank_all.index(kicker_num_1)
                )
            else:
                index += (
                    cards_rank_all.index(kicker_num_2) -
                    cards_rank_all.index(kicker_num_1) + 1
                )
            return index, True
        else:
            return -1, False


def is_quadr2double(card_list):
    # index start from 1
    cards_rank_simple = [
        '3', '4', '5', '6', '7', '8', '9', '10',
        'J', 'Q', 'K', 'A', '2'
    ]
    # NOTE: double could not choose from 'X' and 'D'
    if len(card_list) != 8:
        return -1, False
    else:
        if (
            card_list[0] == card_list[1] == card_list[2] == card_list[3] and
            card_list[4] == card_list[5] and
            card_list[0] != card_list[4] and
            card_list[6] == card_list[7] and
                card_list[0] != card_list[6]):
            main_group_num = card_list[0]
            kicker_num_1 = card_list[6]
            kicker_num_2 = card_list[4]
            # calculate index
            index = cards_rank_simple.index(main_group_num) * 66
            for i in range(0, cards_rank_simple.index(kicker_num_1)):
                index += (11 - i)
            index += (
                cards_rank_simple.index(kicker_num_2) -
                cards_rank_simple.index(kicker_num_1)
            )
            return index, True

        elif (
            card_list[2] == card_list[3] == card_list[4] == card_list[5] and
            card_list[0] == card_list[1] and
            card_list[0] != card_list[2] and
            card_list[6] == card_list[7] and
            card_list[6] != card_list[2] and
                card_list[0] != card_list[6]):
            main_group_num = card_list[2]
            kicker_num_1 = card_list[6]
            kicker_num_2 = card_list[0]
            # calculate index
            index = cards_rank_simple.index(main_group_num) * 66
            for i in range(0, cards_rank_simple.index(kicker_num_1)):
                index += (11 - i)
            index += (
                cards_rank_simple.index(kicker_num_2) -
                cards_rank_simple.index(kicker_num_1) - 1
            )
            return index, True

        elif (
            card_list[4] == card_list[5] == card_list[6] == card_list[7] and
            card_list[0] == card_list[1] and
            card_list[0] != card_list[4] and
            card_list[2] == card_list[3] and
            card_list[2] != card_list[4] and
                card_list[0] != card_list[2]):
            main_group_num = card_list[4]
            kicker_num_1 = card_list[2]
            kicker_num_2 = card_list[0]
            # calculate index
            index = cards_rank_simple.index(main_group_num) * 66
            for i in range(0, cards_rank_simple.index(kicker_num_1)):
                if i < cards_rank_simple.index(main_group_num):
                    index += (11 - i)
                elif i == cards_rank_simple.index(main_group_num):
                    pass
                else:
                    index += (12 - i)
            index += (
                cards_rank_simple.index(kicker_num_2) -
                cards_rank_simple.index(kicker_num_1)
            )
            return index, True

        else:
            return -1, False


def is_planeSingleWing(card_list):
    r""" discuss whether a card_list is a plane with wings(single) and its index
    Plane with Wings(single) need a Trio-Chain to be main group, and
    same amount of single cards as the Trio numbers of the Trio-Chain as wings

    Args:
        card_list: a list of cards
    Returns:
        index: the index of the plane in this kind of combs, starts from 1
        Boolen: Whether this card_list is planeSingleWing
    """
    cards_rank_simple = [
        '3', '4', '5', '6', '7', '8', '9', '10',
        'J', 'Q', 'K', 'A', '2'
    ]
    cards_rank_all = [
        '3', '4', '5', '6', '7', '8', '9', '10',
        'J', 'Q', 'K', 'A', '2', 'X', 'D'
    ]
    if len(card_list) == 8:
        index_1, isTrioChain_1 = is_trioChain(card_list[0:6])
        index_2, isTrioChain_2 = is_trioChain(card_list[1:7])
        index_3, isTrioChain_3 = is_trioChain(card_list[2:])
        if isTrioChain_1:
            # Get the small head of the trio-chain
            plane_small_head = card_list[5]
            # Get the 1st small kicker single wing
            wing_1_single = card_list[7]
            # Get the 2nd small kicker single wing
            wing_2_single = card_list[6]
            index = cards_rank_simple.index(plane_small_head) * 78
            for i in range(0, cards_rank_all.index(wing_1_single)):
                index += (12 - i)
            index += (
                cards_rank_all.index(wing_2_single) -
                cards_rank_all.index(wing_1_single)
            )
            return index, True

        elif isTrioChain_2:
            # Get the small head of the trio-chain
            plane_small_head = card_list[6]
            # Get the 1st small kicker single wing
            wing_1_single = card_list[7]
            # Get the 2nd small kicker single wing
            wing_2_single = card_list[0]
            index = cards_rank_all.index(plane_small_head) * 78
            for i in range(0, cards_rank_all.index(wing_1_single)):
                index += (12 - i)
            index += (
                cards_rank_all.index(wing_2_single) -
                cards_rank_all.index(wing_1_single) - 2
            )
            return index, True

        elif isTrioChain_3:
            # Get the heads of the trio-chain
            plane_small_head = card_list[7]
            plane_big_head = card_list[2]
            # Get the 1st small kicker single wing
            wing_1_single = card_list[1]
            # Get the 2nd small kicker single wing
            wing_2_single = card_list[0]
            index = cards_rank_all.index(plane_small_head) * 78
            for i in range(0, cards_rank_all.index(wing_1_single)):
                if i < cards_rank_all.index(plane_small_head):
                    index += (12 - i)
                elif i > cards_rank_all.index(plane_big_head):
                    index += (14 - i)
                else:
                    pass
            index += (
                cards_rank_all.index(wing_2_single) -
                cards_rank_all.index(wing_1_single)
            )
            return index, True
        else:
            return -1, False

    elif len(card_list) == 12:
        index_1, isTrioChain_1 = is_trioChain(card_list[0:9])
        index_2, isTrioChain_2 = is_trioChain(card_list[1:10])
        index_3, isTrioChain_3 = is_trioChain(card_list[2:11])
        index_4, isTrioChain_4 = is_trioChain(card_list[3:])
        if isTrioChain_1:
            plane_small_head = card_list[8]
            plane_big_head = card_list[0]
            wing_1_single = card_list[11]
            wing_2_single = card_list[10]
            wing_3_single = card_list[9]
            index = 858 + cards_rank_all.index(plane_small_head) * 220
            for i in range(0, cards_rank_all.index(wing_1_single)):
                index += int((11 - i) * (10 - i) / 2)
            for i in range(
                cards_rank_all.index(wing_1_single) + 1,
                    cards_rank_all.index(wing_2_single)):
                index += (15 - 3 - i - 1)
            index += (
                cards_rank_all.index(wing_3_single) -
                cards_rank_all.index(wing_2_single)
            )
            return index, True

        elif isTrioChain_2:
            plane_small_head = card_list[9]
            plane_big_head = card_list[1]
            wing_1_single = card_list[11]
            wing_2_single = card_list[10]
            wing_3_single = card_list[0]
            index = 858 + cards_rank_all.index(plane_small_head) * 220
            for i in range(0, cards_rank_all.index(wing_1_single)):
                index += int((11 - i) * (10 - i) / 2)
            for i in range(
                cards_rank_all.index(wing_1_single) + 1,
                    cards_rank_all.index(wing_2_single)):
                index += (15 - 3 - i - 1)
            index += (
                cards_rank_all.index(wing_3_single) -
                cards_rank_all.index(wing_2_single) - 3
            )
            return index, True

        elif isTrioChain_3:
            plane_small_head = card_list[10]
            plane_big_head = card_list[2]
            wing_1_single = card_list[11]
            wing_2_single = card_list[1]
            wing_3_single = card_list[0]
            index = 858 + cards_rank_all.index(plane_small_head) * 220
            for i in range(0, cards_rank_all.index(wing_1_single)):
                index += int((11 - i) * (10 - i) / 2)
            for i in range(
                cards_rank_all.index(wing_1_single) + 1,
                    cards_rank_all.index(wing_2_single)):
                if i < cards_rank_all.index(plane_small_head):
                    index += (15 - 3 - i - 1)
                elif i > cards_rank_all.index(plane_big_head):
                    index += (15 - i - 1)
                else:
                    pass
            index += (
                cards_rank_all.index(wing_3_single) -
                cards_rank_all.index(wing_2_single)
            )
            return index, True

        elif isTrioChain_4:
            plane_small_head = card_list[11]
            plane_big_head = card_list[3]
            wing_1_single = card_list[2]
            wing_2_single = card_list[1]
            wing_3_single = card_list[0]
            index = 858 + cards_rank_all.index(plane_small_head) * 220
            for i in range(0, cards_rank_all.index(wing_1_single)):
                if i < cards_rank_all.index(plane_small_head):
                    index += int((15 - 3 - i - 1) * (15 - 3 - i - 2) / 2)
                elif i > cards_rank_all.index(plane_big_head):
                    index += int((15 - i - 1) * (15 - i - 2) / 2)
                else:
                    pass
            for i in range(
                cards_rank_all.index(wing_1_single) + 1,
                    cards_rank_all.index(wing_2_single)):
                index += (15 - i - 1)
            index += (
                cards_rank_all.index(wing_3_single) -
                cards_rank_all.index(wing_2_single)
            )
            return index, True

        else:
            return -1, False

    elif len(card_list) == 16:
        index_1, isTrioChain_1 = is_trioChain(card_list[0:12])
        index_2, isTrioChain_2 = is_trioChain(card_list[1:13])
        index_3, isTrioChain_3 = is_trioChain(card_list[2:14])
        index_4, isTrioChain_4 = is_trioChain(card_list[3:15])
        index_5, isTrioChain_5 = is_trioChain(card_list[4:])
        if isTrioChain_1:
            plane_small_head = card_list[11]
            plane_big_head = card_list[0]
            wing_1_single = card_list[15]
            wing_2_single = card_list[14]
            wing_3_single = card_list[13]
            wing_4_single = card_list[12]
            index = 858 + 2200 + cards_rank_all.index(plane_small_head) * 330
            for i in range(0, cards_rank_all.index(wing_1_single)):
                # (15-4-i-1) * (15-4-i-2) * (15-4-i-3) / 3!
                index += int((10 - i) * (9 - i) * (8 - i) / 6)
            for i in range(
                cards_rank_all.index(wing_1_single) + 1,
                    cards_rank_all.index(wing_2_single)):
                index += int((10 - i) * (9 - i) / 2)
            for i in range(
                cards_rank_all.index(wing_2_single) + 1,
                    cards_rank_all.index(wing_3_single)):
                index += (10 - i)
            index += (
                cards_rank_all.index(wing_4_single) -
                cards_rank_all.index(wing_3_single)
            )
            return index, True

        elif isTrioChain_2:
            plane_small_head = card_list[12]
            plane_big_head = card_list[1]
            wing_1_single = card_list[15]
            wing_2_single = card_list[14]
            wing_3_single = card_list[13]
            wing_4_single = card_list[0]
            index = 858 + 2200 + cards_rank_all.index(plane_small_head) * 330
            for i in range(0, cards_rank_all.index(wing_1_single)):
                index += int((10 - i) * (9 - i) * (8 - i) / 6)
            for i in range(
                cards_rank_all.index(wing_1_single) + 1,
                    cards_rank_all.index(wing_2_single)):
                index += int((10 - i) * (9 - i) / 2)
            for i in range(
                cards_rank_all.index(wing_2_single) + 1,
                    cards_rank_all.index(wing_3_single)):
                index += (10 - i)
            index += (
                cards_rank_all.index(wing_4_single) -
                cards_rank_all.index(wing_3_single) - 4
            )
            return index, True

        elif isTrioChain_3:
            plane_small_head = card_list[13]
            plane_big_head = card_list[2]
            wing_1_single = card_list[15]
            wing_2_single = card_list[14]
            wing_3_single = card_list[1]
            wing_4_single = card_list[0]
            index = 858 + 2200 + cards_rank_all.index(plane_small_head) * 330
            for i in range(0, cards_rank_all.index(wing_1_single)):
                index += int((10 - i) * (9 - i) * (8 - i) / 6)
            for i in range(
                cards_rank_all.index(wing_1_single) + 1,
                    cards_rank_all.index(wing_2_single)):
                index += int((10 - i) * (9 - i) / 2)
            for i in range(
                cards_rank_all.index(wing_2_single) + 1,
                    cards_rank_all.index(wing_3_single)):
                if i < cards_rank_all.index(plane_small_head):
                    index += (10 - i)
                elif i > cards_rank_all.index(plane_big_head):
                    index += (14 - i)
                else:
                    pass
            index += (
                cards_rank_all.index(wing_4_single) -
                cards_rank_all.index(wing_3_single)
            )
            return index, True

        elif isTrioChain_4:
            plane_small_head = card_list[14]
            plane_big_head = card_list[3]
            wing_1_single = card_list[15]
            wing_2_single = card_list[2]
            wing_3_single = card_list[1]
            wing_4_single = card_list[0]
            index = 858 + 2200 + cards_rank_all.index(plane_small_head) * 330
            for i in range(0, cards_rank_all.index(wing_1_single)):
                index += int((10 - i) * (9 - i) * (8 - i) / 6)
            for i in range(
                cards_rank_all.index(wing_1_single) + 1,
                    cards_rank_all.index(wing_2_single)):
                if i < cards_rank_all.index(plane_small_head):
                    index += int((10 - i) * (9 - i) / 2)
                elif i > cards_rank_all.index(plane_big_head):
                    index += int((14 - i) * (13 - i) / 2)
                else:
                    pass
            for i in range(
                cards_rank_all.index(wing_2_single) + 1,
                    cards_rank_all.index(wing_3_single)):
                index += (14 - i)
            index += (
                cards_rank_all.index(wing_4_single) -
                cards_rank_all.index(wing_3_single)
            )
            return index, True

        elif isTrioChain_5:
            plane_small_head = card_list[15]
            plane_big_head = card_list[4]
            wing_1_single = card_list[3]
            wing_2_single = card_list[2]
            wing_3_single = card_list[1]
            wing_4_single = card_list[0]
            index = 858 + 2200 + cards_rank_all.index(plane_small_head) * 330
            for i in range(0, cards_rank_all.index(wing_1_single)):
                index += int((10 - i) * (9 - i) * (8 - i) / 6)
                if i < cards_rank_all.index(plane_small_head):
                    index += int((10 - i) * (9 - i) * (8 - i) / 6)
                elif i > cards_rank_all.index(plane_big_head):
                    index += int((14 - i) * (13 - i) * (12 - i) / 6)
                else:
                    pass
            for i in range(
                cards_rank_all.index(wing_1_single) + 1,
                    cards_rank_all.index(wing_2_single)):
                index += int((14 - i) * (13 - i) / 2)
            for i in range(
                cards_rank_all.index(wing_2_single) + 1,
                    cards_rank_all.index(wing_3_single)):
                index += (14 - i)
            index += (
                cards_rank_all.index(wing_4_single) -
                cards_rank_all.index(wing_3_single)
            )
            return index, True

        else:
            return -1, False

    elif len(card_list) == 20:
        index_1, isTrioChain_1 = is_trioChain(card_list[0:15])
        index_2, isTrioChain_2 = is_trioChain(card_list[1:16])
        index_3, isTrioChain_3 = is_trioChain(card_list[2:17])
        index_4, isTrioChain_4 = is_trioChain(card_list[3:18])
        index_5, isTrioChain_5 = is_trioChain(card_list[4:19])
        index_6, isTrioChain_6 = is_trioChain(card_list[5:])
        if isTrioChain_1:
            plane_small_head = card_list[14]
            plane_big_head = card_list[0]
            wing_1_single = card_list[19]
            wing_2_single = card_list[18]
            wing_3_single = card_list[17]
            wing_4_single = card_list[16]
            wing_5_single = card_list[15]
            if (
                not wing_1_single != wing_2_single !=
                    wing_3_single != wing_4_single != wing_5_single):
                return -1, False

            index = 858 + 2200 + 2970
            index += cards_rank_simple.index(plane_small_head) * 252
            for i in range(0, cards_rank_all.index(wing_1_single)):
                # C_(15 - 5 - i - 1)_4
                index += int((9 - i) * (8 - i) * (7 - i) * (6 - i) / 24)
            for i in range(
                cards_rank_all.index(wing_1_single) + 1,
                    cards_rank_all.index(wing_2_single)):
                index += int((9 - i) * (8 - i) * (7 - i) / 6)
            for i in range(
                cards_rank_all.index(wing_2_single) + 1,
                    cards_rank_all.index(wing_3_single)):
                index += int((9 - i) * (8 - i) / 2)
            for i in range(
                cards_rank_all.index(wing_3_single) + 1,
                    cards_rank_all.index(wing_4_single)):
                index += (9 - i)
            index += (
                cards_rank_all.index(wing_5_single) -
                cards_rank_all.index(wing_4_single)
            )
            return index, True
        elif isTrioChain_2:
            plane_small_head = card_list[15]
            plane_big_head = card_list[1]
            wing_1_single = card_list[19]
            wing_2_single = card_list[18]
            wing_3_single = card_list[17]
            wing_4_single = card_list[16]
            wing_5_single = card_list[0]
            index = 858 + 2200 + 2970
            index += cards_rank_simple.index(plane_small_head) * 252
            for i in range(0, cards_rank_all.index(wing_1_single)):
                # C_(15 - 5 - i - 1)_4
                index += int((9 - i) * (8 - i) * (7 - i) * (6 - i) / 24)
            for i in range(
                cards_rank_all.index(wing_1_single) + 1,
                    cards_rank_all.index(wing_2_single)):
                index += int((9 - i) * (8 - i) * (7 - i) / 6)
            for i in range(
                cards_rank_all.index(wing_2_single) + 1,
                    cards_rank_all.index(wing_3_single)):
                index += int((9 - i) * (8 - i) / 2)
            for i in range(
                cards_rank_all.index(wing_3_single) + 1,
                    cards_rank_all.index(wing_4_single)):
                index += (9 - i)
            index += (
                cards_rank_all.index(wing_5_single) -
                cards_rank_all.index(wing_4_single) - 5
            )
            return index, True
        elif isTrioChain_3:
            plane_small_head = card_list[16]
            plane_big_head = card_list[2]
            wing_1_single = card_list[19]
            wing_2_single = card_list[18]
            wing_3_single = card_list[17]
            wing_4_single = card_list[1]
            wing_5_single = card_list[0]
            index = 858 + 2200 + 2970
            index += cards_rank_simple.index(plane_small_head) * 252
            for i in range(0, cards_rank_all.index(wing_1_single)):
                # C_(15 - 5 - i - 1)_4
                index += int((9 - i) * (8 - i) * (7 - i) * (6 - i) / 24)
            for i in range(
                cards_rank_all.index(wing_1_single) + 1,
                    cards_rank_all.index(wing_2_single)):
                index += int((9 - i) * (8 - i) * (7 - i) / 6)
            for i in range(
                cards_rank_all.index(wing_2_single) + 1,
                    cards_rank_all.index(wing_3_single)):
                index += int((9 - i) * (8 - i) / 2)
            for i in range(
                cards_rank_all.index(wing_3_single) + 1,
                    cards_rank_all.index(wing_4_single)):
                if i < cards_rank_all.index(plane_small_head):
                    index += (9 - i)
                elif i > cards_rank_all.index(plane_big_head):
                    index += (14 - i)
                else:
                    pass
            index += (
                cards_rank_all.index(wing_5_single) -
                cards_rank_all.index(wing_4_single)
            )
            return index, True
        elif isTrioChain_4:
            plane_small_head = card_list[17]
            plane_big_head = card_list[3]
            wing_1_single = card_list[19]
            wing_2_single = card_list[18]
            wing_3_single = card_list[2]
            wing_4_single = card_list[1]
            wing_5_single = card_list[0]
            index = 858 + 2200 + 2970
            index += cards_rank_simple.index(plane_small_head) * 252
            for i in range(0, cards_rank_all.index(wing_1_single)):
                # C_(15 - 5 - i - 1)_4
                index += int((9 - i) * (8 - i) * (7 - i) * (6 - i) / 24)
            for i in range(
                cards_rank_all.index(wing_1_single) + 1,
                    cards_rank_all.index(wing_2_single)):
                index += int((9 - i) * (8 - i) * (7 - i) / 6)
            for i in range(
                cards_rank_all.index(wing_2_single) + 1,
                    cards_rank_all.index(wing_3_single)):
                if i < cards_rank_all.index(plane_small_head):
                    index += int((9 - i) * (8 - i) / 2)
                elif i > cards_rank_all.index(plane_big_head):
                    index += int((14 - i) * (13 - i) / 2)
                else:
                    pass
            for i in range(
                cards_rank_all.index(wing_3_single) + 1,
                    cards_rank_all.index(wing_4_single)):
                index += (14 - i)
            index += (
                cards_rank_all.index(wing_5_single) -
                cards_rank_all.index(wing_4_single)
            )
            return index, True
        elif isTrioChain_5:
            plane_small_head = card_list[14]
            plane_big_head = card_list[0]
            wing_1_single = card_list[19]
            wing_2_single = card_list[18]
            wing_3_single = card_list[17]
            wing_4_single = card_list[16]
            wing_5_single = card_list[15]
            index = 858 + 2200 + 2970
            index += cards_rank_simple.index(plane_small_head) * 252
            for i in range(0, cards_rank_all.index(wing_1_single)):
                # C_(15 - 5 - i - 1)_4
                index += int((9 - i) * (8 - i) * (7 - i) * (6 - i) / 24)
            for i in range(
                cards_rank_all.index(wing_1_single) + 1,
                    cards_rank_all.index(wing_2_single)):
                if i < cards_rank_all.index(plane_small_head):
                    index += int((9 - i) * (8 - i) * (7 - i) / 6)
                elif i > cards_rank_all.index(plane_big_head):
                    index += int((14 - i) * (13 - i) * (12 - i) / 6)
                else:
                    pass
            for i in range(
                cards_rank_all.index(wing_2_single) + 1,
                    cards_rank_all.index(wing_3_single)):
                index += int((14 - i) * (13 - i) / 2)
            for i in range(
                cards_rank_all.index(wing_3_single) + 1,
                    cards_rank_all.index(wing_4_single)):
                index += (14 - i)
            index += (
                cards_rank_all.index(wing_5_single) -
                cards_rank_all.index(wing_4_single)
            )
            return index, True
        elif isTrioChain_6:
            plane_small_head = card_list[14]
            plane_big_head = card_list[0]
            wing_1_single = card_list[19]
            wing_2_single = card_list[18]
            wing_3_single = card_list[17]
            wing_4_single = card_list[16]
            wing_5_single = card_list[15]
            index = 858 + 2200 + 2970
            index += cards_rank_simple.index(plane_small_head) * 252
            for i in range(0, cards_rank_all.index(wing_1_single)):
                # C_(15 - 5 - i - 1)_4
                index += int((9 - i) * (8 - i) * (7 - i) * (6 - i) / 24)
                if i < cards_rank_all.index(plane_small_head):
                    index += int((9 - i) * (8 - i) * (7 - i) * (6 - i) / 24)
                elif i > cards_rank_all.index(plane_big_head):
                    index += int(
                        (14 - i) * (13 - i) * (12 - i) * (11 - i) / 24)
                else:
                    pass
            for i in range(
                cards_rank_all.index(wing_1_single) + 1,
                    cards_rank_all.index(wing_2_single)):
                index += int((14 - i) * (13 - i) * (12 - i) / 6)
            for i in range(
                cards_rank_all.index(wing_2_single) + 1,
                    cards_rank_all.index(wing_3_single)):
                index += int((14 - i) * (13 - i) / 2)
            for i in range(
                cards_rank_all.index(wing_3_single) + 1,
                    cards_rank_all.index(wing_4_single)):
                index += (14 - i)
            index += (
                cards_rank_all.index(wing_5_single) -
                cards_rank_all.index(wing_4_single)
            )
            return index, True
        else:
            return -1, False
    else:
        return -1, False


def is_planeDoubleWing(card_list):
    r""" discuss whether a card)_list is a plane with wings(double) and calc index
    Plane with Wings(double) need a Trio-Chain to be main group, and
    same amout of double cards as the Trio num of the Trio-Chain to be Wings.

    Args:
        card_list: a list of cards

    Returns:
        index: the index of the plane in this kind of combs, starts from 1
        Boolen: Whether this card_list is planeDoubleWing
    """
    cards_rank_simple = [
        '3', '4', '5', '6', '7', '8', '9', '10',
        'J', 'Q', 'K', 'A', '2'
    ]
    cards_rank_all = [
        '3', '4', '5', '6', '7', '8', '9', '10',
        'J', 'Q', 'K', 'A', '2', 'X', 'D'
    ]
    if len(card_list) == 10:
        index_1, isTrioChain_1 = is_trioChain(card_list[0:6])
        index_2, isTrioChain_2 = is_trioChain(card_list[2:8])
        index_3, isTrioChain_3 = is_trioChain(card_list[4:10])
        if isTrioChain_1:
            if (
                card_list[9] != card_list[8] or
                card_list[7] != card_list[6] or
                    not card_list[9] != card_list[7]):
                return -1, False
            plane_small_head = card_list[5]
            plane_big_head = card_list[0]
            wing_1_double = card_list[9]
            wing_2_double = card_list[7]
            index = cards_rank_simple.index(plane_small_head) * 55
            for i in range(0, cards_rank_simple.index(wing_1_double)):
                index += (10 - i)
            index += (
                cards_rank_simple.index(wing_2_double) -
                cards_rank_simple.index(wing_1_double)
            )
            return index, True
        elif isTrioChain_2:
            if (
                card_list[9] != card_list[8] or
                    card_list[1] != card_list[0]):
                return -1, False
            plane_small_head = card_list[7]
            plane_big_head = card_list[2]
            wing_1_double = card_list[9]
            wing_2_double = card_list[0]
            index = cards_rank_simple.index(plane_small_head) * 55
            for i in range(0, cards_rank_simple.index(wing_1_double)):
                index += (10 - i)
            index += (
                cards_rank_all.index(wing_2_double) -
                cards_rank_all.index(wing_1_double) - 2
            )
            return index, True
        elif isTrioChain_3:
            if (
                card_list[3] != card_list[2] or
                card_list[1] != card_list[0] or
                    not card_list[3] != card_list[1]):
                return -1, False
            plane_small_head = card_list[9]
            plane_big_head = card_list[4]
            wing_1_double = card_list[3]
            wing_2_double = card_list[1]
            index = cards_rank_simple.index(plane_small_head) * 55
            for i in range(0, cards_rank_simple.index(wing_1_double)):
                if i < cards_rank_simple.index(plane_small_head):
                    index += (10 - i)
                elif i > cards_rank_simple.index(plane_big_head):
                    index += (12 - i)
                else:
                    pass
            index = (
                cards_rank_simple.index(wing_2_double) -
                cards_rank_simple.index(wing_1_double)
            )
            return index, True
        return -1, False

    elif len(card_list) == 15:
        index_1, isTrioChain_1 = is_trioChain(card_list[0:9])
        index_2, isTrioChain_2 = is_trioChain(card_list[2:11])
        index_3, isTrioChain_3 = is_trioChain(card_list[4:13])
        index_4, isTrioChain_4 = is_trioChain(card_list[6:15])
        if isTrioChain_1:
            plane_small_head = card_list[8]
            plane_big_head = card_list[0]
            wing_1_double = card_list[13]
            wing_2_double = card_list[11]
            wing_3_double = card_list[9]
            index = 605 + cards_rank_simple.index(plane_small_head) * 120
            for i in range(0, cards_rank_simple.index(wing_1_double)):
                # C_(13-3-i-1)_2
                index += int((9 - i) * (8 - i) / 2)
            for i in range(
                cards_rank_simple.index(wing_1_double) + 1,
                    cards_rank_simple.index(wing_2_double)):
                index += (9 - i)
            index += (
                cards_rank_simple.index(wing_3_double) -
                cards_rank_simple.index(wing_2_double)
            )
            return index, True
        elif isTrioChain_2:
            plane_small_head = card_list[10]
            plane_big_head = card_list[2]
            wing_1_double = card_list[13]
            wing_2_double = card_list[11]
            wing_3_double = card_list[0]
            index = 605 + cards_rank_simple.index(plane_small_head) * 120
            for i in range(0, cards_rank_simple.index(wing_1_double)):
                index += int((9 - i) * (8 - i) / 2)
            for i in range(
                cards_rank_simple.index(wing_1_double) + 1,
                    cards_rank_simple.index(wing_2_double)):
                index += (9 - i)
            index += (
                cards_rank_simple.index(wing_3_double) -
                cards_rank_simple.index(wing_2_double) - 3
            )
            return index, True
        elif isTrioChain_3:
            plane_small_head = card_list[12]
            plane_big_head = card_list[4]
            wing_1_double = card_list[13]
            wing_2_double = card_list[2]
            wing_3_double = card_list[0]
            index = 605 + cards_rank_simple.index(plane_small_head) * 120
            for i in range(0, cards_rank_simple.index(wing_1_double)):
                index += int((9 - i) * (8 - i) / 2)
            for i in range(
                cards_rank_simple.index(wing_1_double) + 1,
                    cards_rank_simple.index(wing_2_double)):
                if i < cards_rank_simple.index(plane_small_head):
                    index += (13 - 3 - i - 1)
                elif i > cards_rank_simple.index(plane_big_head):
                    index += (13 - i - 1)
                else:
                    pass
            index += (
                cards_rank_simple.index(wing_3_double) -
                cards_rank_simple.index(wing_2_double)
            )
            return index, True
        elif isTrioChain_4:
            plane_small_head = card_list[14]
            plane_big_head = card_list[6]
            wing_1_double = card_list[4]
            wing_2_double = card_list[2]
            wing_3_double = card_list[0]
            index = 605 + cards_rank_simple.index(plane_small_head) * 120
            for i in range(0, cards_rank_simple.index(wing_1_double)):
                if i < cards_rank_simple.index(plane_small_head):
                    index += int((13 - 3 - i - 1) * (13 - 3 - i - 2) / 2)
                elif i > cards_rank_simple.index(plane_big_head):
                    index += int((13 - i - 1) * (13 - i - 2) / 2)
                else:
                    pass
            for i in range(
                cards_rank_simple.index(wing_1_double) + 1,
                    cards_rank_simple.index(wing_2_double)):
                index += (13 - i - 1)
            index += (
                cards_rank_simple.index(wing_3_double) -
                cards_rank_simple.index(wing_2_double)
            )
            return index, True
        else:
            return -1, False
    elif len(card_list) == 20:
        index_1, isTrioChain_1 = is_trioChain(card_list[0:12])
        index_2, isTrioChain_2 = is_trioChain(card_list[2:14])
        index_3, isTrioChain_3 = is_trioChain(card_list[4:16])
        index_4, isTrioChain_4 = is_trioChain(card_list[6:18])
        index_5, isTrioChain_5 = is_trioChain(card_list[8:20])
        if isTrioChain_1:
            plane_small_head = card_list[11]
            plane_big_head = card_list[0]
            wing_1_double = card_list[18]
            wing_2_double = card_list[16]
            wing_3_double = card_list[14]
            wing_4_double = card_list[12]
            # 1805 = 605 + 1200
            index = 1805 + cards_rank_simple.index(plane_small_head) * 126
            for i in range(0, cards_rank_simple.index(wing_1_double)):
                # C_(13-4-i-1)_3
                index += int((8 - i) * (7 - i) * (6 - i) / 6)
            for i in range(
                cards_rank_simple.index(wing_1_double) + 1,
                    cards_rank_simple.index(wing_2_double)):
                index += int((8 - i) * (7 - i) / 2)
            for i in range(
                cards_rank_simple.index(wing_2_double) + 1,
                    cards_rank_simple.index(wing_3_double)):
                index += (8 - i)
            index += (
                cards_rank_simple.index(wing_4_double) -
                cards_rank_simple.index(wing_3_double)
            )
            return index, True

        elif isTrioChain_2:
            plane_small_head = card_list[13]
            plane_big_head = card_list[2]
            wing_1_double = card_list[18]
            wing_2_double = card_list[16]
            wing_3_double = card_list[14]
            wing_4_double = card_list[0]
            index = 1805 + cards_rank_simple.index(plane_small_head) * 126
            for i in range(0, cards_rank_simple.index(wing_1_double)):
                # C_(13-4-i-1)_3
                index += int((8 - i) * (7 - i) * (6 - i) / 6)
            for i in range(
                cards_rank_simple.index(wing_1_double) + 1,
                    cards_rank_simple.index(wing_2_double)):
                index += int((8 - i) * (7 - i) / 2)
            for i in range(
                cards_rank_simple.index(wing_2_double) + 1,
                    cards_rank_simple.index(wing_3_double)):
                index += (8 - i)
            index += (
                cards_rank_simple.index(wing_4_double) -
                cards_rank_simple.index(wing_3_double) - 4
            )
            return index, True

        elif isTrioChain_3:
            plane_small_head = card_list[15]
            plane_big_head = card_list[4]
            wing_1_double = card_list[18]
            wing_2_double = card_list[16]
            wing_3_double = card_list[2]
            wing_4_double = card_list[0]
            index = 1805 + cards_rank_simple.index(plane_small_head) * 126
            for i in range(0, cards_rank_simple.index(wing_1_double)):
                # C_(13-4-i-1)_3
                index += int((8 - i) * (7 - i) * (6 - i) / 6)
            for i in range(
                cards_rank_simple.index(wing_1_double) + 1,
                    cards_rank_simple.index(wing_2_double)):
                index += int((8 - i) * (7 - i) / 2)
            for i in range(
                cards_rank_simple.index(wing_2_double) + 1,
                    cards_rank_simple.index(wing_3_double)):
                if i < cards_rank_simple.index(plane_small_head):
                    index += (8 - i)
                elif i > cards_rank_simple.index(plane_big_head):
                    index += (12 - i)
                else:
                    pass
            index += (
                cards_rank_simple.index(wing_4_double) -
                cards_rank_simple.index(wing_3_double)
            )
            return index, True
        elif isTrioChain_4:
            plane_small_head = card_list[17]
            plane_big_head = card_list[6]
            wing_1_double = card_list[18]
            wing_2_double = card_list[4]
            wing_3_double = card_list[2]
            wing_4_double = card_list[0]
            index = 1805 + cards_rank_simple.index(plane_small_head) * 126
            for i in range(0, cards_rank_simple.index(wing_1_double)):
                # C_(13-4-i-1)_3
                index += int((8 - i) * (7 - i) * (6 - i) / 6)
            for i in range(
                cards_rank_simple.index(wing_1_double) + 1,
                    cards_rank_simple.index(wing_2_double)):
                if i < cards_rank_simple.index(plane_small_head):
                    index += int((8 - i) * (7 - i) / 2)
                elif i > cards_rank_simple.index(plane_big_head):
                    index += int((12 - i) * (11 - i) / 2)
                else:
                    pass
            for i in range(
                cards_rank_simple.index(wing_2_double) + 1,
                    cards_rank_simple.index(wing_3_double)):
                index += (12 - i)
            index += (
                cards_rank_simple.index(wing_4_double) -
                cards_rank_simple.index(wing_3_double)
            )
            return index, True
        elif isTrioChain_5:
            plane_small_head = card_list[19]
            plane_big_head = card_list[8]
            wing_1_double = card_list[6]
            wing_2_double = card_list[4]
            wing_3_double = card_list[2]
            wing_4_double = card_list[0]
            index = 1805 + cards_rank_simple.index(plane_small_head) * 126
            for i in range(0, cards_rank_simple.index(wing_1_double)):
                if i < cards_rank_simple.index(plane_small_head):
                    index += int((8 - i) * (7 - i) * (6 - i) / 6)
                elif i > cards_rank_simple.index(plane_big_head):
                    index += int((12 - i) * (11 - i) * (10 - i) / 6)
                else:
                    pass
            for i in range(
                cards_rank_simple.index(wing_1_double) + 1,
                    cards_rank_simple.index(wing_2_double)):
                index += int((12 - i) * (11 - i) / 2)
            for i in range(
                cards_rank_simple.index(wing_2_double) + 1,
                    cards_rank_simple.index(wing_3_double)):
                index += (12 - i)
            index += (
                cards_rank_simple.index(wing_4_double) -
                cards_rank_simple.index(wing_3_double)
            )
            return index, True
        else:
            return -1, False


def label_str2int(label_str):
    r""" Generate the int-style card_combs from str-comb
    The total elements of card_combs is 13707 (contains PASS).
    When training in neural networks, the label and output of the NN
    should be one-hot tensors. Thus, when doing training, the output
    needed to be coverted from str to int (index), one-hot is not necessary.

    Args:
        label_str: the str-style of the specific cards_comb

    Returns:
        index: the int index (0 - 13550) of that specific cards_comb
        index_cur: the int index among this category of combs
        descip: the kind of combs
    """

    cards_rank_simp = [
        '3', '4', '5', '6', '7', '8', '9', '10',
        'J', 'Q', 'K', 'A', '2'
    ]
    cards_rank_all = [
        '3', '4', '5', '6', '7', '8', '9', '10',
        'J', 'Q', 'K', 'A', '2', 'X', 'D'
    ]

    label_list = split_handcards(label_str)

    if label_str == 'P':
        # Pass
        # num_Pass = 1
        return 0, 0, 'Pass'

    elif label_str == 'DX' or label_str == 'XD':
        # Rocket
        # num_Rocket = 1
        return 1, 0, 'Rocket'

    elif len(label_list) == 1 and label_list[0] in cards_rank_all:
        # Single
        # num_Single = 15
        index = cards_rank_all.index(label_list[0])
        # num_Pass + num_Rocket
        return 2 + index, index, 'Single'

    elif len(label_list) == 2 and label_list[0] == label_list[1]:
        # Double
        if label_list[0] not in cards_rank_simp:
            raise ValueError(
                'Double should be among 3-2, got {}'.format(label_str)
            )
        else:
            # num_Double = 13
            index = cards_rank_simp.index(label_list[0])
            # num_Pass + num_Rocket + num_Single
            return 17 + index, index, 'Double'

    elif (len(label_list) == 3 and
            label_list[0] == label_list[1] == label_list[2]):
        # Trio
        if label_list[0] not in cards_rank_simp:
            raise ValueError(
                'Trio should be among 3-2, got {}'.format(label_str)
            )
        else:
            # num_Trio = 13
            index = cards_rank_simp.index(label_list[0])
            # num_Pass + num_Rocket + num_Single + num_Double
            return 30 + index, index, 'Trio'

    elif (len(label_list) == 4 and
            label_list[0] == label_list[1] == label_list[2] == label_list[3]):
        # Bomb
        if label_list[0] not in cards_rank_simp:
            raise ValueError(
                'Bomb should be among 3-2, got {}'.format(label_str)
            )
        else:
            # num_Bomb = 13
            index = cards_rank_simp.index(label_list[0])
            # sigma previous number
            return 43 + index, index, 'Bomb'

    elif (len(label_list) == 4 and
            (label_list[0] == label_list[1] == label_list[2] or
                label_list[1] == label_list[2] == label_list[3])):
        # Trio + Single, total num: 182
        main_group_num = ''
        kicker_single_num = ''
        # The four-cards' rank is descending
        if label_list[0] == label_list[1] == label_list[2]:
            # kicker < main group
            if label_list[0] not in cards_rank_simp:
                raise ValueError(
                    'Trio Main Group should be among 3-2, got {}'
                    .format(label_str)
                )
            else:
                main_group_num = label_list[0]
                kicker_single_num = label_list[3]
                # calculate two-parts' index
                index_main = cards_rank_simp.index(main_group_num)
                index_kicker = cards_rank_all.index(kicker_single_num)
                return (
                    56 + index_main * 14 + index_kicker,
                    index_main * 14 + index_kicker,  'Trio1Single')

        elif label_list[1] == label_list[2] == label_list[3]:
            # kicker > main group
            if label_list[1] not in cards_rank_simp:
                raise ValueError(
                    'Trio Main Group should be among 3-2, got {}'
                    .format(label_str)
                )
            else:
                main_group_num = label_list[1]
                kicker_single_num = label_list[0]
                # calculate two-part's index
                index_main = cards_rank_simp.index(main_group_num)
                index_kicker = cards_rank_all.index(kicker_single_num)
                return (
                    56 + index_main * 14 + index_kicker - 1,
                    index_main * 14 + index_kicker - 1, 'Trio1Single')

        else:
            raise ValueError(
                'Need Descending sort Pokers, got {}'
                .format(label_str)
            )

    elif (
            len(label_list) == 5 and
            (
                (
                    label_list[0] == label_list[1] == label_list[2] and
                    label_list[3] == label_list[4]) or
                (
                    label_list[2] == label_list[3] == label_list[4] and
                    label_list[0] == label_list[1]))):
        # Trio + Double, total num: 156
        main_group_num = ''
        kicker_double_num = ''
        # The five-cards' rank is descending
        if label_list[0] == label_list[1] == label_list[2]:
            # kicker < main group
            if label_list[0] not in cards_rank_simp:
                raise ValueError(
                    'Trio Main Group should be among 3-2, got {}'
                    .format(label_str)
                )
            elif label_list[3] not in cards_rank_simp:
                raise ValueError(
                    'Trio Kicker Group should be among 3-2, got {}'
                    .format(label_str)
                )
            else:
                main_group_num = label_list[0]
                kicker_double_num = label_list[3]
                # calculate two-parts' index
                index_main = cards_rank_simp.index(main_group_num)
                index_kicker = cards_rank_simp.index(kicker_double_num)
                return (
                    238 + index_main * 12 + index_kicker,
                    index_main * 12 + index_kicker, 'Trio1Double')

        elif label_list[2] == label_list[3] == label_list[4]:
            # kicker > main group
            if label_list[2] not in cards_rank_simp:
                raise ValueError(
                    'Trio Main Group should be among 3-2, got {}'
                    .format(label_str)
                )
            elif label_list[0] not in cards_rank_simp:
                raise ValueError(
                    'Trio Kicker Group should be among 3-2, got {}'
                    .format(label_str)
                )
            else:
                main_group_num = label_list[2]
                kicker_double_num = label_list[0]
                # calculate two-part's index
                index_main = cards_rank_simp.index(main_group_num)
                index_kicker = cards_rank_simp.index(kicker_double_num)
                return (
                    238 + index_main * 12 + index_kicker - 1,
                    index_main * 12 + index_kicker - 1, 'Trio1Double')

    else:
        indexSingleChain, isSingleChain = is_singleChain(label_list)
        if isSingleChain:
            return 394 + indexSingleChain, indexSingleChain, 'SingleChain'

        indexDoubleChain, isDoubleChain = is_doubleChain(label_list)
        if isDoubleChain:
            return 430 + indexDoubleChain, indexDoubleChain, 'DoubleChain'

        indexTrioChain, isTrioChain = is_trioChain(label_list)
        if isTrioChain:
            return 482 + indexTrioChain, indexTrioChain, 'TrioChain'

        indexQuadr2Single, isQuadr2Single = is_quadr2single(label_list)
        if isQuadr2Single:
            return (
                527 + indexQuadr2Single - 1,
                indexQuadr2Single - 1, 'Quadr2Single')

        indexQuadr2Double, isQuadr2Double = is_quadr2double(label_list)
        if isQuadr2Double:
            return (
                1866 + indexQuadr2Double - 1,
                indexQuadr2Double - 1, 'Quadr2Double')

        (indexPlaneSingleWing,
            isPlaneSingleWing) = is_planeSingleWing(label_list)
        if isPlaneSingleWing:
            return (
                2724 + indexPlaneSingleWing - 1,
                indexPlaneSingleWing - 1, 'PlaneSingleWing')

        (indexPlaneDoubleWing,
            isPlaneDoubleWing) = is_planeDoubleWing(label_list)
        if isPlaneDoubleWing:
            return (
                10768 + indexPlaneDoubleWing - 1,
                indexPlaneDoubleWing - 1, 'PlaneDoubleWing')

        else:
            raise ValueError(
                'cards comb mismatched! got {}'
                .format(label_str)
            )


def label_int2str(cards_int):
    r""" Generate the str-style card_combs from int index
    The total elements of card_combs is 13707 (contains PASS).
    When training in neural networks, the label and output of the NN
    should be one-hot tensors. Thus, when doing inference, the output
    needed to be coverted from one-hot tensor (or just index) to str.

    Args:
        cards_int: the index(int) of the specific cards_comb

    Returns:
        cards_str: the str-style of that specific cards_comb
    """
    # TODO: implement this!!!
    return str(cards_int)


def generate_game_process(
        landlord, landlord_down, landlord_up,
        public, game_process, game_winner):
    r""" Generate Game State before each play of the winner

    Args:
        landlord: list of init handcards
        landlord_down: list of init handcards
        landlord_up: list of init handcards
        public: list of init public cards
        game_process:
        game_winner:

    Return:
        steps_data: list of 3-dim numpy array
        steps_label: list of labels (string)
        steps_label_index: list of labels (int)
    """

    steps_data = []
    steps_label = []
    steps_label_index = []

    # temporary multi-state-features
    landlord_public = public

    landlord_played = []
    landlord_down_played = []
    landlord_up_played = []

    landlord_last_played = []
    landlord_down_last_played = []
    landlord_up_last_played = []

    # Only the winner's handcard could be known, Otherwise remain empty
    landlord_handcard = []
    landlord_down_handcard = []
    landlord_up_handcard = []

    # Whether if the winner is landlord / landlord_down / landlord_up

    # What cards haven't been played

    # if there is a trio in the winner's handcard

    if game_winner == '0':
        # NOTE: landlord's handcard should also contain public cards
        landlord_handcard = landlord + public
    elif game_winner == '1':
        landlord_down_handcard = landlord_down
    elif game_winner == '2':
        landlord_up_handcard = landlord_up
    else:
        raise ValueError(
            'game winner can only be (char)0, 1, 2, got {}'
            .format(game_winner)
        )

    # Game Process for each step
    (landlord_steps, landlord_down_steps,
        landlord_up_steps) = game_process_with_pass(game_process)

    # check whether the PASS added accurately
    if game_winner == '0':
        if len(landlord_steps) != len(landlord_down_steps) + 1:
            raise ValueError('generated steps with PASS has incorrect size')
        elif len(landlord_down_steps) != len(landlord_up_steps):
            raise ValueError('generated steps with PASS has incorrect size')
    elif game_winner == '1':
        if len(landlord_down_steps) != len(landlord_up_steps) + 1:
            raise ValueError('generated steps with PASS has incorrect size')
        elif len(landlord_steps) != len(landlord_down_steps):
            raise ValueError('generated steps with PASS has incorrect size')
    elif game_winner == '2':
        if len(landlord_steps) != len(landlord_steps):
            raise ValueError('generated steps with PASS has incorrect size')
        elif len(landlord_steps) != len(landlord_down_steps):
            raise ValueError('generated steps with PASS has incorrect size')

    if game_winner == '0':
        for i in range(0, len(landlord_steps)):
            plane_0 = cards_rank_encode(landlord_public)
            plane_1 = cards_rank_encode(landlord_played)
            plane_2 = cards_rank_encode(landlord_down_played)
            plane_3 = cards_rank_encode(landlord_up_played)
            plane_4 = cards_rank_encode(landlord_last_played)
            plane_5 = cards_rank_encode(landlord_down_last_played)
            plane_6 = cards_rank_encode(landlord_up_last_played)
            plane_7 = cards_rank_encode(landlord_handcard)
            plane_8 = cards_rank_encode(landlord_down_handcard)
            plane_9 = cards_rank_encode(landlord_up_handcard)

            # stack planes -> C * H * W
            step_data = np.stack(
                (plane_0, plane_1, plane_2, plane_3, plane_4,
                    plane_5, plane_6, plane_7, plane_8, plane_9), axis=0
            )
            # Get winner's current playing cards as label
            step_label = landlord_steps[i]
            # print('process: {}'.format(step_label))
            step_label_index, _, _ = label_str2int(step_label)
            steps_data.append(step_data)
            steps_label.append(step_label)
            steps_label_index.append(step_label_index)

            # Check whether the game is end or not
            if i == len(landlord_steps) - 1:
                pass
            else:
                # player's last played cards
                # NOTE: Here I also put PASS into the cards played records
                landlord_played.extend(
                    split_handcards(landlord_steps[i])
                )
                landlord_down_played.extend(
                    split_handcards(landlord_down_steps[i])
                )
                landlord_up_played.extend(
                    split_handcards(landlord_up_steps[i])
                )
                landlord_last_played = split_handcards(
                    landlord_steps[i]
                )
                landlord_down_last_played = split_handcards(
                    landlord_down_steps[i]
                )
                landlord_up_last_played = split_handcards(
                    landlord_up_steps[i]
                )

                # calculate landlord's current handcards
                for elem in landlord_last_played:
                    if elem != 'P':
                        landlord_handcard.remove(elem)

    elif game_winner == '1':
        # landlord should have played one step before player '1'
        landlord_played = split_handcards(landlord_steps[0])
        landlord_last_played = split_handcards(landlord_steps[0])
        # As game_winner is '1', I couldn't know landlord's handcard

        for i in range(0, len(landlord_down_steps)):
            plane_0 = cards_rank_encode(landlord_public)
            plane_1 = cards_rank_encode(landlord_played)
            plane_2 = cards_rank_encode(landlord_down_played)
            plane_3 = cards_rank_encode(landlord_up_played)
            plane_4 = cards_rank_encode(landlord_last_played)
            plane_5 = cards_rank_encode(landlord_down_last_played)
            plane_6 = cards_rank_encode(landlord_up_last_played)
            plane_7 = cards_rank_encode(landlord_handcard)
            plane_8 = cards_rank_encode(landlord_down_handcard)
            plane_9 = cards_rank_encode(landlord_up_handcard)

            # stack planes -> C * H * W
            step_data = np.stack(
                (plane_0, plane_1, plane_2, plane_3, plane_4,
                    plane_5, plane_6, plane_7, plane_8, plane_9), axis=0
            )
            # Get winner's current playing cards as label
            step_label = landlord_down_steps[i]
            step_label_index, _, _ = label_str2int(step_label)
            steps_data.append(step_data)
            steps_label.append(step_label)
            steps_label_index.append(step_label_index)

            # check whether the game is end or not
            if i == len(landlord_down_steps) - 1:
                pass
            else:
                # player's last played cards
                # NOTE: Here I also put PASS into the cards played records
                landlord_played.extend(
                    split_handcards(landlord_steps[i + 1])
                )
                landlord_down_played.extend(
                    split_handcards(landlord_down_steps[i])
                )
                landlord_up_played.extend(
                    split_handcards(landlord_up_steps[i])
                )
                landlord_last_played = split_handcards(
                    landlord_steps[i + 1]
                )
                landlord_down_last_played = split_handcards(
                    landlord_down_steps[i]
                )
                landlord_up_last_played = split_handcards(
                    landlord_up_steps[i]
                )

                # calculate landlord_down's current handcards
                for elem in landlord_down_last_played:
                    if elem != 'P':
                        landlord_down_handcard.remove(elem)

    elif game_winner == '2':
        # landlord should have played one step before player '1'
        landlord_played = split_handcards(landlord_steps[0])
        landlord_last_played = split_handcards(landlord_steps[0])
        # As game_winner is '2', I couldn't know landlord's handcard

        # landlord_down should have played one step before player '2'
        landlord_down_played = split_handcards(landlord_down_steps[0])
        landlord_down_last_played = split_handcards(landlord_down_steps[0])
        # As game_winner is '2', I couldn't know landlord_down's handcard

        for i in range(0, len(landlord_up_steps)):
            plane_0 = cards_rank_encode(landlord_public)
            plane_1 = cards_rank_encode(landlord_played)
            plane_2 = cards_rank_encode(landlord_down_played)
            plane_3 = cards_rank_encode(landlord_up_played)
            plane_4 = cards_rank_encode(landlord_last_played)
            plane_5 = cards_rank_encode(landlord_down_last_played)
            plane_6 = cards_rank_encode(landlord_up_last_played)
            plane_7 = cards_rank_encode(landlord_handcard)
            plane_8 = cards_rank_encode(landlord_down_handcard)
            plane_9 = cards_rank_encode(landlord_up_handcard)

            # stack planes -> C * H * W
            step_data = np.stack(
                (plane_0, plane_1, plane_2, plane_3, plane_4,
                    plane_5, plane_6, plane_7, plane_8, plane_9), axis=0
            )
            # Get winner's current playing cards as label
            step_label = landlord_up_steps[i]
            step_label_index, _, _ = label_str2int(step_label)
            steps_data.append(step_data)
            steps_label.append(step_label)
            steps_label_index.append(step_label_index)

            # check whether the game is end or not
            if i == len(landlord_up_steps) - 1:
                pass
            else:
                # player's last played cards
                # NOTE: Here I also put PASS into the cards played records
                landlord_played.extend(
                    split_handcards(landlord_steps[i + 1])
                )
                landlord_down_played.extend(
                    split_handcards(landlord_down_steps[i + 1])
                )
                landlord_up_played.extend(
                    split_handcards(landlord_up_steps[i])
                )
                landlord_last_played = split_handcards(
                    landlord_steps[i + 1]
                )
                landlord_down_last_played = split_handcards(
                    landlord_down_steps[i + 1]
                )
                landlord_up_last_played = split_handcards(
                    landlord_up_steps[i]
                )

                # calculate landlord_up's current handcards
                for elem in landlord_up_last_played:
                    if elem != 'P':
                        landlord_up_handcard.remove(elem)

    return steps_data, steps_label, steps_label_index


if __name__ == "__main__":
    r""" Main Function of DataLoader
    Need to make Unit-Test for the cards_comb's str2int part
    """

    with open(opt.inputFile, 'rt') as f_1:
        cnt_line = 0
        cnt_npy = 1

        np_array_data = None
        np_array_label = None
        np_array_flag = False
        np_array_data_left = None
        np_array_label_left = None

        for line in f_1:
            if cnt_line == opt.train_num:
                break

            cnt_line += 1
            # Got Game Process
            cards = line.split(' Game process:')[0]
            # Got cards parts
            cards = cards.strip('Cards:')

            # Got Game Process
            game_process = line.split(' Game process:')[1].strip('\n')

            # split four parts of the cards records
            cards_landlord = cards.split(';')[0]
            cards_landlord_down = cards.split(';')[1]
            cards_landlord_up = cards.split(';')[2]
            cards_landlord_public = cards.split(';')[-1]

            # split string structures of the card series into seperate list
            cards_landlord = split_handcards(cards_landlord)
            cards_landlord_down = split_handcards(cards_landlord_down)
            cards_landlord_up = split_handcards(cards_landlord_up)
            cards_landlord_public = split_handcards(cards_landlord_public)

            # convert list to binary numpy array
            cards_landlord_array = \
                cards_rank_encode(cards_landlord)
            cards_landlord_down_array = \
                cards_rank_encode(cards_landlord_down)
            cards_landlord_up_array = \
                cards_rank_encode(cards_landlord_up)
            cards_landlord_public_array = \
                cards_rank_encode(cards_landlord_public)

            # Add Pass to the Game Process
            (landlord_game, landlord_down_game,
                landlord_up_game) = game_process_with_pass(game_process)

            all_data, all_label, all_label_index = generate_game_process(
                cards_landlord, cards_landlord_down, cards_landlord_up,
                cards_landlord_public, game_process, str(opt.personID)
            )

            if not np_array_flag:
                # Read a new line of game process after reach or exceed 500
                # or fresh start
                if np_array_data_left is None:
                    np_array_data = np.stack(all_data, axis=0)
                    np_array_label = np.stack(all_label_index, axis=0)
                else:
                    current_data = np.stack(all_data, axis=0)
                    current_label = np.stack(all_label_index, axis=0)
                    np_array_data = np.concatenate(
                        (np_array_data_left, current_data), axis=0
                    )
                    np_array_label = np.concatenate(
                        (np_array_label_left, current_label), axis=0
                    )
                    np_array_data_left = None
                    np_array_label_left = None

                np_array_flag = True

            else:
                current_data = np.stack(all_data, axis=0)
                current_label = np.stack(all_label_index, axis=0)

                if np_array_data.shape[0] + current_data.shape[0] > 500:
                    overflow_length = (
                        np_array_data.shape[0] + current_data.shape[0] - 500)
                    concat_length = current_data.shape[0] - overflow_length

                    np_array_data = np.concatenate(
                        (np_array_data, current_data[0:concat_length]),
                        axis=0
                    )
                    np_array_label = np.concatenate(
                        (np_array_label, current_label[0:concat_length]),
                        axis=0
                    )

                    print(
                        'save {} piece of data. '
                        'State Shape: {}, Label Shape: {}'
                        .format(
                            cnt_npy, np_array_data.shape, np_array_label.shape)
                    )
                    np.save(
                        os.path.join(
                            opt.save_dir, 'data', 'all_state_%d' % cnt_npy),
                        np_array_data)
                    np.save(
                        os.path.join(
                            opt.save_dir, 'label', 'all_label_%d' % cnt_npy),
                        np_array_label)
                    cnt_npy += 1

                    np_array_data_left = current_data[concat_length:]
                    np_array_label_left = current_label[concat_length:]
                    np_array_data = None
                    np_array_label = None

                    np_array_flag = False

                elif np_array_data.shape[0] + current_data.shape[0] == 500:
                    # save to .npy file, clear buffer
                    np_array_data = np.concatenate(
                        (np_array_data, current_data), axis=0
                    )
                    np_array_label = np.concatenate(
                        (np_array_label, current_label), axis=0
                    )

                    print(
                        'save {} piece of data. '
                        'State Shape: {}, Label Shape: {}'
                        .format(
                            cnt_npy, np_array_data.shape, np_array_label.shape)
                    )
                    np.save(
                        os.path.join(
                            opt.save_dir, 'data', 'all_state_%d' % cnt_npy),
                        np_array_data)
                    np.save(
                        os.path.join(
                            opt.save_dir, 'label', 'all_label_%d' % cnt_npy),
                        np_array_label)
                    cnt_npy += 1

                    np_array_data_left = None
                    np_array_label_left = None
                    np_array_data = None
                    np_array_label = None

                    np_array_flag = False

                else:
                    # concat, keep moving
                    np_array_data = np.concatenate(
                        (np_array_data, current_data), axis=0
                    )
                    np_array_label = np.concatenate(
                        (np_array_label, current_label), axis=0
                    )

        if np_array_flag:
            if np_array_data is None and np_array_data_left is not None:
                print(
                    'save {} piece of data. '
                    'State Shape: {}, Label Shape: {}'
                    .format(
                        cnt_npy, np_array_data_left.shape,
                        np_array_label_left.shape)
                )
                np.save(
                    os.path.join(
                        opt.save_dir, 'data', 'all_state_%d' % cnt_npy),
                    np_array_data_left)
                np.save(
                    os.path.join(
                        opt.save_dir, 'label', 'all_label_%d' % cnt_npy),
                    np_array_label_left)
            elif np_array_data is not None:
                print(
                    'save {} piece of data. '
                    'State Shape: {}, Label Shape: {}'
                    .format(
                        cnt_npy, np_array_data.shape, np_array_label.shape)
                )
                np.save(
                    os.path.join(
                        opt.save_dir, 'data', 'all_state_%d' % cnt_npy),
                    np_array_data)
                np.save(
                    os.path.join(
                        opt.save_dir, 'label', 'all_label_%d' % cnt_npy),
                    np_array_label)

        print('Finished! Total Records: {}'.format(cnt_line))
