import neat
import numpy as np
import subprocess
import select 
import pickle

class CustomReporter(neat.reporting.BaseReporter):
    def __init__(self):
        self.f = open("generation_fitness_log.txt", "w")  # Open a file for writing

    def post_generation(self, config, population, species_set, generation):
        # Log generation number and fitness scores
        self.f.write(f"Generation {generation}\n")
        for genome_id, genome in population.items():
            self.f.write(f"Genome ID: {genome_id}, Fitness: {genome.fitness}\n")
        self.f.write("\n")  # Add newline for separation

    def finish_generation(self, config, population, species_set, generation):
        pass  # Optional: perform any cleanup or finalization steps

    def close(self):
        self.f.close()  # Close the file when finished



def play_superball(net):
    superball_path = "./sb-player"
    with open('tmp.fitness', 'w') as f:
        f.write('0')
    command = [superball_path, '8', '10', '1', 'pbyrg', './run_net.sh', 'n', 'n', '-']  # Correct the command to run the subprocess
    with subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) as process:
        lines = [line for line in process.stdout]
        print(lines)
        cleaned_text = lines[0].split()[-1]     

        #print(cleaned_text[0])
        with open('tmp.fitness', 'r') as f:
            lines = f.readline()
            words = lines.split()
            bad_moves = int(words[0])
            print("printing bad moves")
            print(bad_moves)
            score = int(cleaned_text)
            fitnesstmp = (score+1) * (100/(bad_moves+1))
            print(fitnesstmp)
        return fitnesstmp


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        with open(f"genome.pkl", "wb") as f:
            pickle.dump(genome, f)
        avg_score = np.average([play_superball(net) for _ in range(2)])        
        with open(f"generations.tx", "a") as f:
            f.write(f"{genome_id}: {avg_score}\n")
        genome.fitness = avg_score

def main():
    config_file = './config_file'
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_file)
    p = neat.Population(config)

    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(5))

    winner = p.run(eval_genomes, 500)
    with open('tmp.test', 'a') as f:
        f.write('\nBest genome:\n{!s}'.format(winner))


if  __name__ == '__main__':
	main()
