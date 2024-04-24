import neat
import pickle
import sys
from disjoint_sets import DisjointSets


def default_swap(board):
    x = 0
    y = 0
    for i, val in enumerate(board):
        if val != '.' and val != '*':
            x = i
            break
    for j, val in enumerate(board):
        if val != '*' and val != '.' and j != x:
            y = j
            break
    
    print(f"SWAP {int(x/10)} {x%10} {int(y/10)} {y%10}")
def main():
    with open('genome.pkl', 'rb') as f:
        genome = pickle.load(f)
        config_file = './config_file'
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_file)
        net = neat.nn.FeedForwardNetwork.create(genome, config)

        net_input = [char for line in sys.stdin for char in line.strip()]
        net_input = [ord(i) for i in net_input]
        no = net.activate(net_input)
        no = [int(i) for i in no]
        no[1] = 0 if no[1] < 0 else (7 if no[1] > 7 else no[1])
        no[2] = 0 if no[2] < 0 else (9 if no[2] > 9 else no[2])
        no[3] = 0 if no[3] < 0 else (7 if no[3] > 7 else no[3])
        no[4] = 0 if no[4] < 0 else (9 if no[4] > 9 else no[4])

        game_input = f"SWAP {no[1]} {no[2]} {no[3]} {no[4]}" if no[0] > 0 else f"SCORE {no[1]} {no[2]}"
        if no[0] > 0 and (chr(net_input[no[1]*10+no[2]]) in ['*', '.'] or chr(net_input[no[3]*10+no[4]]) in ['*','.']):
            net_input = [chr(i) for i in net_input]
            default_swap(net_input)
            return
        elif no[0] <= 0:
            scoring_range = [20, 21, 28, 29, 30, 31, 38, 39, 40, 41, 48, 49, 50, 51, 58, 59]
            if chr(net_input[no[1]*10+no[2]]) in ['*','.'] or no[1]*10+no[2] not in scoring_range:
                net_input = [chr(i) for i in net_input]
                default_swap(net_input)
                return

        ds = DisjointSets()
        for i in range(len(net_input)):
            if chr(net_input[i]) in ['*','.']:
                continue
            if i % 10 != 9:
                if net_input[i] == net_input[i+1]:
                    ds.union(i, i+1)
            if int(i/10) != 7:
                if net_input[i] == net_input[i+10]:
                    ds.union(i, i+10)

            if ds.get_set_size(i) < 5:
                net_input = [chr(i) for i in net_input]
                default_swap(net_input)
                return
        print(game_input)

if  __name__ == '__main__':
    main()

