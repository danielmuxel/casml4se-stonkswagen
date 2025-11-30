
from experiment import train
import itertools


params = {
    "hidden_size": [32],
    "num_layers": [3,8],
    "num_attention_heads": [2,8],
    "lookback": [20],
    "horizon": [5],
    "dropout": [0],
    "layer_dropout": [0],
    "learning_rate": [0.01],
    "weight_decay": [1e-4],
    "batch_size": [128],
    "use_conv": [True, False],
    "conv_kernel_size": [3,6,12]
}

# Calculate total number of combinations
total_runs = 1
for param_values in params.values():
    total_runs *= len(param_values)

print(f"Total training runs to execute: {total_runs}\n")

current_run = 0

for hidden_size in params["hidden_size"]:
    for num_layers in params["num_layers"]:
        for num_attention_heads in params["num_attention_heads"]:
            for lookback in params["lookback"]:
                for horizon in params["horizon"]:
                    for dropout in params["dropout"]:
                        for layer_dropout in params["layer_dropout"]:
                            for learning_rate in params["learning_rate"]:
                                for weight_decay in params["weight_decay"]:
                                    for batch_size in params["batch_size"]:
                                        for use_conv in params["use_conv"]:
                                            for conv_kernel_size in params["conv_kernel_size"]:

                                                current_run += 1
                                                print(f"\n{'=' * 60}")
                                                print(f"Running experiment {current_run}/{total_runs}")
                                                print(f"{'=' * 60}\n")

                                                train(
                                                    model_type='advanced',
                                                    hidden_size=hidden_size,
                                                    num_layers=num_layers,
                                                    num_attention_heads=num_attention_heads,
                                                    lookback=lookback,
                                                    horizon=horizon,
                                                    dropout=dropout,
                                                    layer_dropout=layer_dropout,
                                                    learning_rate=learning_rate,
                                                    weight_decay=weight_decay,
                                                    batch_size=batch_size,
                                                    use_conv=use_conv,
                                                    epochs=20,
                                                    conv_kernel_size=conv_kernel_size,
                                                    data_path="/home/lukas/Documents/Github/casml4se-stonkswagen/data/cryptos/bitcoin-tiny.csv",
                                                    run_name=f'run{current_run}/{total_runs}_h{hidden_size}_l{num_layers}_ah{num_attention_heads}_lb{lookback}_hz{horizon}_d{dropout}_ld{layer_dropout}_lr{learning_rate}_wd{weight_decay}_bs{batch_size}_conv{use_conv}_ck{conv_kernel_size}'
                                                )
