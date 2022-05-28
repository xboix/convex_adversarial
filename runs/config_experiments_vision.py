import json
import os


def config_experiments(results_dir, create_json=True):

    with open('./base_config.json') as config_file:
        base_config = json.load(config_file)

    id = 0
    experiment_list = []
    for dataset, dataset_id in zip(['mnist'], [0]):
        for net in ["ThreeLayer"]:
            for batch_size in [32, 256]:
                for normalization in ["01", "standarized"]:

                    restart = False

                    if normalization == "standarized":
                        standarize = True
                        multiplier = 255.0
                        upper = 10e10
                        lower = -10e10
                    else:
                        standarize = False
                        multiplier = 1.0
                        upper = 1.0
                        lower = 0.0

                    #Vanilla
                    for lr in [1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]:
                        config = base_config.copy()
                        config["data_set"] = dataset
                        config["data_set_id"] = dataset_id
                        config["model_name"] = str(id)
                        config["restart"] = restart
                        config["backbone"] = net
                        config["training_batch_size"] = batch_size
                        config["initial_learning_rate"] = lr
                        config["robust_training"] = False
                        config["pgd_training"] = False
                        config["epsilon_scheduler"] = False
                        config["max_num_training_steps"] = 10000
                        config["batch_decrease_learning_rate"] = 1e10  # do not decrease the learning rate
                        config["bound_lower"] = lower
                        config["bound_upper"] = upper
                        config["standarize"] = standarize
                        config["standarize_multiplier"] = multiplier

                        if create_json:
                            with open(results_dir + 'configs/' + str(id)+'.json', 'w') as json_file:
                                json.dump(config, json_file)
                        experiment_list.append(config.copy())
                        id += 1

                    #Convex approx
                    for lr in [1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]:
                        for epsilon in [1e-4, 1e-5, 1e-3, 1e-2, 1e-1, 3e-1, 5e-1, 1, 3, 5, 10]:
                            config = base_config.copy()
                            config["data_set"] = dataset
                            config["data_set_id"] = dataset_id
                            config["model_name"] = str(id)
                            config["restart"] = restart
                            config["training_batch_size"] = batch_size
                            config["backbone"] = net
                            config["initial_learning_rate"] = lr
                            config["epsilon"] = epsilon
                            config["max_num_training_steps"] = 10000
                            config["robust_training"] = True
                            config["pgd_training"] = False
                            config["epsilon_scheduler"] = False
                            config["batch_decrease_learning_rate"] = 1e10  # do not decrease the learning rate
                            config["bound_lower"] = lower
                            config["bound_upper"] = upper
                            config["standarize"] = standarize
                            config["standarize_multiplier"] = multiplier

                            if create_json:
                                with open(results_dir + 'configs/' + str(id)+'.json', 'w') as json_file:
                                    json.dump(config, json_file)
                            experiment_list.append(config.copy())
                            id += 1

                    #Convex approx scheduler
                    for lr in [1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]:
                        for epsilon in [1e-4, 1e-5, 1e-3, 1e-2, 1e-1, 3e-1, 5e-1, 1, 3, 5, 10]:
                            config = base_config.copy()
                            config["data_set"] = dataset
                            config["data_set_id"] = dataset_id
                            config["model_name"] = str(id)
                            config["restart"] = restart
                            config["training_batch_size"] = batch_size
                            config["backbone"] = net
                            config["initial_learning_rate"] = lr
                            config["epsilon"] = epsilon
                            config["max_num_training_steps"] = 10000
                            config["robust_training"] = True
                            config["pgd_training"] = False
                            config["epsilon_scheduler"] = True
                            config["batch_decrease_learning_rate"] = 1e10  # do not decrease the learning rate
                            config["bound_lower"] = lower
                            config["bound_upper"] = upper
                            config["standarize"] = standarize
                            config["standarize_multiplier"] = multiplier

                            if create_json:
                                with open(results_dir + 'configs/' + str(id)+'.json', 'w') as json_file:
                                    json.dump(config, json_file)
                            experiment_list.append(config.copy())
                            id += 1


    print(str(id) + " config files created")
    return experiment_list


def check_uncompleted(results_dir, experiments_list):

    for experiment in experiments_list:
        if experiment["skip"]:
            continue
        if not os.path.isfile(results_dir + experiment["model_name"] + '/results/training.done'):
            print(experiment["model_name"], end = ',')

    print("\n Check train completed")

