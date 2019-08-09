# -*- coding: utf-8 -*-
import sys

sys.path.append('../')

import torch
import numpy as np
from game.engine import Agent, Game, GameState, Card
from data.DataLoader import label_int2str
import model.resnet as resnet
import argparse

parser = argparse.ArgumentParser(
    description='ArgParser for the eval on engine.')
parser.add_argument(
    '-d',
    '--device',
    type=str,
    default='cuda:0',
    help='using GPU and cuda to train and also indicate the GPU number'
)
parser.add_argument(
    '-sd',
    '--seed',
    type=int,
    help='indicate the seed number while doing random to reach reproductable'
)
parser.add_argument(
    '--model_path',
    type=str,
    default='../expr/best_acc.pth',
    help='path to the model'
)
args = parser.parse_args()
print(args)


class RandomModel(Agent):
    def choose(self, state):
        # get all valid moves and randomly choose one
        valid_moves = self.get_moves()
        i = np.random.choice(len(valid_moves))
        move = valid_moves[i]

        # player i [手牌] // [出牌]
        hand_card = []
        for i, n in enumerate(Card.all_card_name):
            hand_card.extend([n] * self.get_hand_card()[i])

        print(
            "Player {}"
            .format(self.player_id),
            ' ', hand_card, ' // ', Card.visual_card(move)
        )

        return move, None


def findByRow(mat, row):
    return np.where((mat == row).all(1))[0]


def cards_rank_encode(cards):
    r""" Cards rank number encoder
    Convert a card rank list into binary numpy array

    Args:
        cards: A list of card ranks
        cards: A 15-len numpy array. EXAMPLE: array([1,2,3,...,0])

    Output:
        A numpy array which only contain 0 or 1
        and the size of this array is 15 * 4
    """
    # NOTE. Here we are using the bool type of numpy array
    binary_array = np.zeros((15, 4), dtype=bool)
    # card_ranks = [
    #     '3', '4', '5', '6', '7', '8', '9',
    #     '10', 'J', 'Q', 'K', 'A', '2', 'X', 'D']

    # for card in cards:
    #     if card != 'P':
    #         index = card_ranks.index(card)
    #         for i in range(0, 4):
    #             if binary_array[index][i]:
    #                 pass
    #             else:
    #                 binary_array[index][i] = 1
    #                 break

    for i in range(15):
        for j in range(cards[i]):
            binary_array[i][j] = 1

    return binary_array


def generate_game_state(
        handcards, public, game_process, player_id):
    r""" Generate Game State before action made by the CNN Agent

    Args:
        landlord: numpy array len of 15 (remain [] is unknown)
        landlord_down: numpy array len of 15 (remain [] if unknown)
        landlord_up: numpy array len of 15 (remain [] if unknown)
        public: numpy array len of 15
        game_process: series of moves until now
        player_id: 0,1,2 which seperately indicated landlord / down / up

    Returns:
        state_data: current state
    """

    state_data = None

    # temporary multi-state-features
    landlord_public = public
    landlord_played = np.zeros(15, dtype=int)
    landlord_down_played = np.zeros(15, dtype=int)
    landlord_up_played = np.zeros(15, dtype=int)
    landlord_last_played = np.zeros(15, dtype=int)
    landlord_down_last_played = np.zeros(15, dtype=int)
    landlord_up_last_played = np.zeros(15, dtype=int)
    landlord_handcard = np.zeros(15, dtype=int)
    landlord_down_handcard = np.zeros(15, dtype=int)
    landlord_up_handcard = np.zeros(15, dtype=int)

    if player_id == 0:
        # NOTE: landlord's handcard should also contain public cards
        landlord_handcard = handcards
    elif player_id == 1:
        landlord_down_handcard = handcards
    elif player_id == 2:
        landlord_up_handcard = handcards
    else:
        raise ValueError(
            'player id can only be (int) 0, 1, 2. Got: {}'
            .format(player_id)
        )

    # Process for each game step
    # The input of game log should be sth such as:
    # [(0,array([1,0,1,...,0])),(1,array([0,...,0])),(2,...),...]
    for out in game_process:
        cur_player = out[0]
        cur_cards_out = out[1]
        # Check whether the move is in the type of numpy array
        if isinstance(cur_cards_out, np.ndarray):
            # DO NOTHING
            pass
        elif (
            isinstance(cur_cards_out, list) and
                isinstance(cur_cards_out[0], int)):
            # convert list-type of moves into np.ndarray
            cur_cards_out = np.array(cur_cards_out)
        # Seperatelly consider to add the play history for diff platers
        if cur_player == 0:
            landlord_played += cur_cards_out
            landlord_last_played = cur_cards_out
        elif cur_player == 1:
            landlord_down_played += cur_cards_out
            landlord_down_last_played = cur_cards_out
        elif cur_player == 2:
            landlord_up_played += cur_cards_out
            landlord_up_last_played = cur_cards_out
        else:
            raise ValueError(
                'player id can only be (int) 0, 1, 2. Got: {}'
                .format(cur_player)
            )

    # convert all these 15-len nunpy array to binary numpy 15 * 4 array
    # Calling *cards_rank_encode*
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

    # stack these planes
    state_data = np.stack(
        (plane_0, plane_1, plane_2, plane_3, plane_4,
            plane_5, plane_6, plane_7, plane_8, plane_9), axis=0
    )

    return state_data


class CNNModel(Agent):
    r""" Class of the Model. Inherit from Agent
    description:
        self.player_id: 当前玩家id, 0:地主, 1:地主下家, 2:地主上家
        self.get_hand_card():当前剩余手牌，格式: 长度15 numpy array [1,0,0,...,0]
        self.game.cards_out:所有玩家按序打出的牌。格式: [(player_id, move), ... ]
        self.game_last_desc:上家出的牌的描述(如果上家未出则是上上家)
                            三元组(总张数,主牌rank,类型)
        self.game_last_move:上家出的牌
        self.get_public_card():获得地主明牌，格式: 长度15 numpy array [1,1,1,...,0]

        应返回出牌列表，空列表表示不要/要不起， 格式举例：
            [13]
            [1,1,1,4]
            [3,3]
            [14,15]
            [2,2,2,2]
    """
    def choose(self, state):
        # state -(CNNModel)-> card_combs
        current_handcards = self.get_hand_card()
        series_cards_out = self.game.cards_out
        player_id = self.player_id
        # Process Public
        public_np = self.get_public_card()
        # calc binary np array state (15*4 bool)
        state = generate_game_state(
            current_handcards, public_np, series_cards_out, player_id)

        if self.player_id == 0:
            # Using the model trained on Landlord
            valid_moves = self.get_moves()
            # Load Trained Net
            net = resnet.resnetpokernet(num_classes=13707).to(device)
            net.load_state_dict(torch.load(args.model_path))
            batch_state = np.expand_dims(state, axis=0)
            batch_state_ = batch_state.to(device)
            outputs = net(batch_state_)
            _, pred = torch.topk(outputs, 10)

            if isinstance(valid_moves, np.ndarray):
                pass
            else:
                valid_moves = np.array(valid_moves)

            for i in range(10):
                _, list_idx = label_int2str(pred[0][i].item())
                find_res = findByRow(valid_moves, np.array(list_idx))
                if len(find_res) == 1:
                    move = valid_moves[find_res]
                    hand_card = []
                    for j, n in enumerate(Card.all_card_name):
                        hand_card.extend([n] * self.get_hand_card()[j])
                    # player i [手牌] // [出牌]
                    print(
                        "Player {}"
                        .format(self.player_id),
                        ' ', hand_card, ' // ', Card.visual_card(move)
                    )
                    return move, None

            # Don't exist in top-10, return PASS
            # rand_i = np.random.choice(len(valid_moves))
            return [], None

        elif self.player_id == 1:
            # Using the model trained on Landlord_down
            # Would be loaded later
            pass
        elif self.player_id == 2:
            # Using the model trained on Landlord_up
            # Would be loaded later
            pass
        else:
            raise ValueError(
                'player_id should be among 0, 1, 2. Got: {}'
                .format()
            )
        return []


if __name__ == "__main__":
    if args.device is not None:
        if args.device.startswith('cuda') and torch.cuda.is_available():
            device = torch.device(args.device)
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            device = torch.device('cpu')
            # logger.warning(
            #     '{} GPU is not available, running on CPU'.format(__name__))
            print(
                'Warning: {} GPU is not available, running on CPU'
                .format(__name__))
            torch.set_default_tensor_type('torch.FloatTensor')
    else:
        if torch.cuda.is_available():
            device = torch.device('cuda')
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            device = torch.device('cpu')

    print('{} Using device: {}'.format(__name__, device))

    # Seeding
    if args.seed is not None:
        # logger.info('{} Setting random seed'.format(__name__))
        print('{} Setting random seed'.format(__name__))
        seed = args.seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

    # game = Game([RandomModel(i) for i in range(3)])
    # game should be a combine of Agents to form a Game
    game = Game([CNNModel(0)] + [RandomModel(i) for i in range(1, 3)])
    MAX_ROUNDS = 100
    TRAIND_ID = 0	    # 进行训练的模型，0代表地主，1代表地主下家，2代表地主上家

    for i_episode in range(1):
        game.game_reset()
        game.show()
        for i in range(MAX_ROUNDS):
            pid, state, cur_moves, cur_move, winner, info = game.step()
            game.show()
            if winner != -1:
                print('Winner: {}'.format(winner))
                if TRAIND_ID == 0 and winner == 0:
                    # do some positive reward
                    pass
                elif TRAIND_ID != 0 and winner != 0:
                    # do some positive reward
                    pass
                else:
                    # do some negative reward
                    pass
                break
