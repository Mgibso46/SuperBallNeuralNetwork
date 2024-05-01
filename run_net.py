import neat
import pickle
import sys
import math
from scipy.cluster.hierarchy import DisjointSet


def default_swap(board):
    x = 0
    y = 0            
    with open('tmp.fitness', 'r') as f:
            line = f.readline()
            words = line.split()
            bad_moves = words[0]
    with open('tmp.fitness', 'w') as f:
        f.write(f"{int(bad_moves)+1}")

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

        net_input = [char.lower() for line in sys.stdin for char in line.strip()]
        net_input = [ord(i) for i in net_input]
        no = net.activate(net_input)
        #with open('tmp.output', 'a') as f:
        #    f.write(f"{no[0]} {no[1]} {no[2]} {no[3]} {no[4]} | ")
        
        no = [math.floor(float(i)*8)%8 for i in no]
        no[0] = no[0]%2
        #with open('tmp.output', 'a') as f:
        #    f.write(f"{no[0]} {no[1]} {no[2]} {no[3]} {no[4]} | ")
        #no[1] = 0 if no[1] < 0 else (7 if no[1] > 7 else no[1])
        #no[2] = 0 if no[2] < 0 else (9 if no[2] > 9 else no[2])
        #no[3] = 0 if no[3] < 0 else (7 if no[3] > 7 else no[3])
        #no[4] = 0 if no[4] < 0 else (9 if no[4] > 9 else no[4])

        game_input = f"SWAP {no[1]} {no[2]} {no[3]} {no[4]}" if no[0] > 0 else f"SCORE {no[1]} {no[2]}"
        #with open('tmp.output', 'a') as f:
        #    f.write(f"{game_input}\n")
            
        i = no[1]*10+no[2]
        j = no[3]*10+no[4]
        #with open('tmp.output', 'a') as f:
        #    f.write(f"{no[1]} {i} {chr(net_input[i])} {(chr(net_input[i]) in ['*', '.'])} {j} {chr(net_input[j])} {chr(net_input[j]) in ['*','.']}\n")

        if no[0] == 1:
            with open('swapCount.tmp', 'r') as f:
                line = f.readline()
                words = line.split()
                swapA = words[0]
            with open(f"swapCount.tmp", "w") as f:
                f.write(f"{int(swapA)+1}")
        elif no[0] == 0:        
            with open('scoreCount.tmp', 'r') as f:
                line = f.readline()
                words = line.split()
                scoreA = words[0]
            with open(f"scoreCount.tmp", 'w') as f:
                f.write(f"{int(scoreA)+1}")
        if no[0] == 1 and (chr(net_input[i]) in ['*', '.'] or chr(net_input[j]) in ['*','.'] or i == j):
            #with open('tmp.output', 'a') as f:
            #    f.write(f"default swap on SWAP\n")

            net_input = [chr(i) for i in net_input]
            default_swap(net_input)
            return
        elif no[0] == 0:
            #with open('tmp.output', 'a') as f:
            #    f.write(f"default swap on score\n")
            scoring_range = [20, 21, 28, 29, 30, 31, 38, 39, 40, 41, 48, 49, 50, 51, 58, 59]
            index = no[1]*10+no[2]
            #with open('tmp.output', 'a') as f:
            #    f.write(f"INDEX = {index}\n")
            if chr(net_input[index]) == '*' or index not in scoring_range:
            #    with open('tmp.output', 'a') as f:
            #        f.write(f"default swap on SCORE ZONE\n")
                net_input = [chr(i) for i in net_input]
                default_swap(net_input)

                return

            ds = DisjointSet([i for i in range(len(net_input)) if chr(net_input[i]) not in ['*', ',']])
            for i in range(len(net_input)):
                if chr(net_input[i]) in ['*','.']:
                    continue
                row = i // 10
                col = i % 10
                if col < 9 and net_input[i] == net_input[i + 1]:
                    ds.merge(i, i+1)
                if row < 7 and net_input[i] == net_input[i + 10]:
                    ds.merge(i, i+10)
            if len(ds.subset(no[1]*10+no[2])) < 2:
                net_input = [chr(i) for i in net_input]
                default_swap(net_input)
                return
        print(game_input)

if  __name__ == '__main__':
    main()

