import neat
import numpy as np
import subprocess
import select 
import pickle

def play_superball(net):
    superball_path = "./sb-player"
    command = [superball_path, '8', '10', '1', 'pbyrg', './run_net.sh', 'n', 'n', '-']  # Correct the command to run the subprocess
    with subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) as process:
        for line1 in process.stdout:
            line2 = line1.split()
            print(line2, end = '')
            cleaned_text = line2[-1]

        #print(cleaned_text[0])
        with open('tmp.fitness', 'r') as f:
            lines = f.readline()
            words = lines.split()
            bad_moves = int(words[0])
            print("printing bad moves")
            print(bad_moves)
            score = int(cleaned_text[0])
            print((score+1) * (100/(bad_moves+1)))
        return (score+1) * (100/(bad_moves+1))


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        with open(f"genome.pkl", "wb") as f:
            pickle.dump(genome, f)
        avg_score = np.average([play_superball(net) for _ in range(2)])        
        genome.fitness = avg_score

def main():
    config_file = './config_file'
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_file)
    p = neat.Population(config)

    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(5))

    winner = p.run(eval_genomes, 10)

    print('\nBest genome:\n{!s}'.format(winner))


if  __name__ == '__main__':
	main()
