from experiment import train
import itertools


params = {
    "hidden_size": [64, 128],
    "num_layers": [1, 2, 3],
    "lookback": [10,20,50],
    "horizon": [5],
    "dropout": [0],
    "learning_rate": [0.001],
    "batch_size": [64, 128]
}

# Calculate total number of combinations
total_runs = 1
for param_values in params.values():
    total_runs *= len(param_values)

print(f"Total training runs to execute: {total_runs}\n")

current_run = 0

for hidden_size in params["hidden_size"]:
    for num_layers in params["num_layers"]:
        for lookback in params["lookback"]:
            for horizon in params["horizon"]:
                for dropout in params["dropout"]:
                    for learning_rate in params["learning_rate"]:
                        for batch_size in params["batch_size"]:

                            current_run += 1
                            print(f"\n{'=' * 60}")
                            print(f"Running experiment {current_run}/{total_runs}")
                            print(f"{'=' * 60}\n")

                            train(
                                model_type='simple',
                                hidden_size=hidden_size,
                                num_layers=num_layers,
                                lookback=lookback,
                                horizon=horizon,
                                dropout=dropout,
                                learning_rate=learning_rate,
                                batch_size=batch_size,
                                epochs=20,
                                data_path="/home/lukas/Documents/Github/casml4se-stonkswagen/data/cryptos/bitcoin-tiny.csv",
                                run_name=f'run{current_run}/{total_runs}_h{hidden_size}_l{num_layers}_lb{lookback}_hz{horizon}_d{dropout}_lr{learning_rate}_bs{batch_size}'
                            )
