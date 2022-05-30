Code to compare [Provably robust neural networks (convex_adversarial)](https://github.com/locuslab/convex_adversarial) and [A Robust Optimization approach to deep learning (RUB)](https://github.com/kimvc7/Robustness/). This repo has been forked fom [convex_adversarial](https://github.com/locuslab/convex_adversarial) and adapted to run a hyperparameter search and the same testing as in [RUB](https://github.com/kimvc7/Robustness/), for a fair comparison. 

The code executes as in RUB while using the functions provided in convex_adversarial (see both original repos for more details).


## Requirements 

Docker needs to be installed in your system. Pull the following docker container:
```
docker pull xboixbosch/tf
```
Then, run bash in the container and execute all the code there.
```
docker run -v <CODE PATH>:/home/neuro/Robustness -it xboixbosch/tf bash
```

## Preparing and running the experiments

1. Generate the experiment configuration files:

`runs/config_experiment_vision.py` generate one configuration file per network to train and evaluate.
To generate all the configuration files run the following:
```
python3 main.py --run=config --config=generate
```
These command will create ~1K json files for the vision datasets. Each file
describes the network, hyperparameters, dataset, etc. of an experiment. The name of the file is a number that corresponds
to the `experiment_id`.

3. Run the training and testing:

To train, test, and evaluate the bound use the following commands:
```
python3 main.py --run=train --experiment_id=<experiment_id> --gpu_id=0
python3 main.py --run=test --experiment_id=<experiment_id> --gpu_id=0
```
where `experiment_id` corresponds to the number of the experiment and `gpu_id` indicates the ID of the GPUs to run use 
(the current version of the code does not support multi-GPU).

4. Analyze the results:

Use the jupter notebooks in `notebooks` folder. `Vision.ipynb` do the cross-validation for each attack and rho and save all relevant information in a csv file. 


