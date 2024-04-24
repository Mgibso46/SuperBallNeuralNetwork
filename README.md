# SuperBallNeuralNetwork

## Documentation
Here is the documentation for [NEAT](https://neat-python.readthedocs.io/en/latest/xor_example.html)

## Important Files
* `run_net.py` - plays `sb-player` with the net currently saved in `genome.pkl`
* `test.py` - Uses NEAT to train net to play superball. Uses the configuration in `config_file`
* `run_net.sh` - script used to run `run_net.py` in the superball player
* `sb-player` - Dr. planks superball executable

## Installation
```bash
pip install -r requirements.txt
```
To run:
```bash
python test.py
```

## What needs to be changed
To tune the GA and Net, change the settings in the config file.  
To change the fitness function modify `eval_genome` in `test.py`.

##### NOTE: THERE MAY STILL BE BUGS and test.py is just a starting file it will need updates for sure.
