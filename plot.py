import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

def main(args: argparse.Namespace):
    assert os.path.exists(args.logs_path), "Invalid logs path"

    # for i in [True, False]:
    #     for j in range(1, 6):
    #         assert os.path.exists(os.path.join(args.logs_path, f"run_{j}_{i}.npy")),\
    #             f"File run_{j}_{i}.npy not found in {args.logs_path}"
    # TODO: Load data and plot the standard means and standard deviations of
    # the accuracies for the two settings (active and random strategies)
    # TODO: also ensure that the files have the same length
    
    # Load training accuracy data for both Active Learning Strategy(ALS) and Random Strategy(RS)
    active_train_runs = []
    random_train_runs = []
    
    # Load validation accuracy data for both Active Learning Strategy(ALS) and Random Strategy(RS)          
    active_val_runs = []
    random_val_runs = []
    
    # Load timing data for original re-training and modified updates
    original_times_run = []  # Store timing data for original training
    modified_times_run = []  # Store timing data for modified incremental updates


    for run_id in range(1, 6):
        # Load training accuracy data
        active_train_path = os.path.join(args.logs_path, f"train_run_{run_id}_True.npy")
        random_train_path = os.path.join(args.logs_path, f"train_run_{run_id}_False.npy")
        assert os.path.exists(active_train_path), f"File {active_train_path} not found!"
        assert os.path.exists(random_train_path), f"File {random_train_path} not found!"
        active_train_runs.append(np.load(active_train_path))
        random_train_runs.append(np.load(random_train_path))
        
        # Load validation accuracy data
        active_val_path = os.path.join(args.logs_path, f"run_{run_id}_True.npy")
        random_val_path = os.path.join(args.logs_path, f"run_{run_id}_False.npy")
        assert os.path.exists(active_val_path), f"File {active_val_path} not found!"
        assert os.path.exists(random_val_path), f"File {random_val_path} not found!"
        active_val_runs.append(np.load(active_val_path))
        random_val_runs.append(np.load(random_val_path))

        # Load timing data
        original_times_path = os.path.join(args.logs_path, f"original_times_{run_id}.npy")
        modified_times_path = os.path.join(args.logs_path, f"modified_times_{run_id}.npy")
        assert os.path.exists(original_times_path), f"File {original_times_path} not found!"
        assert os.path.exists(modified_times_path), f"File {modified_times_path} not found!"
        original_times_run.append(np.load(original_times_path))
        modified_times_run.append(np.load(modified_times_path))


    # Convert lists to NumPy arrays
    active_train_runs = np.array(active_train_runs)
    random_train_runs = np.array(random_train_runs)
     
    active_val_runs = np.array(active_val_runs)
    random_val_runs = np.array(random_val_runs)
    
    original_times_run = np.array(original_times_run)
    modified_times_run = np.array(modified_times_run)  


    # Ensure all runs have the same length
    assert active_train_runs.shape == random_train_runs.shape, "Mismatch in run lengths!"
    assert active_train_runs.shape == active_val_runs.shape, "Mismatch in run lengths!"
    assert active_train_runs.shape == random_val_runs.shape, "Mismatch in run lengths!"
    assert active_train_runs.shape == original_times_run.shape, "Mismatch in run lengths!"
    assert active_train_runs.shape == modified_times_run.shape, "Mismatch in run lengths!"
    
    
    # Compute Mean and Standard Deviation for accuracy
    # Set max x-axis range to 50,000 for comparison
    max_x_points = 50000  
    no_of_iterations = active_val_runs.shape[1]              # Ensure it matches dataset length
    step_size = max_x_points // no_of_iterations             # Calculate step size for x-axis scaling
    x_axis = np.arange(1, no_of_iterations + 1) * step_size  # Generate x-axis values


    # Compute mean and standard deviation for both Active Learning Strategy(ALS) and Random Strategy(RS)
    active_train_mean = np.mean(active_train_runs, axis=0)
    active_train_std = np.std(active_train_runs, axis=0)
    
    random_train_mean = np.mean(random_train_runs, axis=0)
    random_train_std = np.std(random_train_runs, axis=0)
    
    active_val_mean = np.mean(active_val_runs, axis=0)
    active_val_std = np.std(active_val_runs, axis=0)
    
    random_val_mean = np.mean(random_val_runs, axis=0)
    random_val_std = np.std(random_val_runs, axis=0)


    # Compute Mean and Standard Deviation for timings
    original_times_mean = np.mean(original_times_run, axis=0)
    original_times_std = np.std(original_times_run, axis=0)

    modified_times_mean = np.mean(modified_times_run, axis=0)
    modified_times_std = np.std(modified_times_run, axis=0)

    '''# Find the index where ALS first reaches supervised accuracy
    als_reach_idx = np.argmax(active_train_mean >= args.supervised_accuracy)  
    rs_reach_idx = np.argmax(random_train_mean >= args.supervised_accuracy)  

    # Convert index to actual number of labeled data points
    als_data_required = x_axis[als_reach_idx]
    rs_data_required = x_axis[rs_reach_idx]

    # Compute percentage reduction
    data_reduction = ((rs_data_required - als_data_required) / rs_data_required) * 100

    print(f"Data reduction using Active Learning: {data_reduction:.2f}%")'''
    # Find the index where ALS first reaches supervised accuracy
    als_reach_idx = np.where(active_train_mean >= args.supervised_accuracy)[0]
    rs_reach_idx = np.where(random_train_mean >= args.supervised_accuracy)[0]

    # Check if ALS and RS reached the supervised accuracy
    if len(als_reach_idx) == 0:
        print("ALS never reached the supervised accuracy.")
        als_data_required = None
    else:
        als_data_required = x_axis[als_reach_idx[0]]  # First index where ALS reaches the accuracy

    if len(rs_reach_idx) == 0:
        print("Random Sampling never reached the supervised accuracy.")
        rs_data_required = None
    else:
        rs_data_required = x_axis[rs_reach_idx[0]]  # First index where RS reaches the accuracy

    # Compute percentage reduction only if both ALS and RS reached the accuracy
    if als_data_required is not None and rs_data_required is not None:
        data_reduction = ((rs_data_required - als_data_required) / rs_data_required) * 100
        print(f"Data reduction using Active Learning: {data_reduction:.2f}%")
    else:
        print("Cannot compute data reduction because one or both methods did not reach the supervised accuracy.")   



#################################################################### Plot #####################################################################
    
    fig, axes = plt.subplots(1, 2, figsize=(17, 5))  # 1 row, 2 columns


    # ---- Subplot 1: Accuracy Plot (Training & Validation) ----
    axes[0].plot(x_axis, active_train_mean, label="Train (ALS)", color="blue", linestyle="dashed", linewidth=2)
    axes[0].fill_between(x_axis, active_train_mean - active_train_std, active_train_mean + active_train_std, alpha=0.2, color="blue")

    axes[0].plot(x_axis, active_val_mean, label="Validation (ALS)", color="blue", linewidth=2)
    axes[0].fill_between(x_axis, active_val_mean - active_val_std, active_val_mean + active_val_std, alpha=0.2, color="blue")

    axes[0].plot(x_axis, random_train_mean, label="Train (RS)", color="red", linestyle="dashed", linewidth=2)
    axes[0].fill_between(x_axis, random_train_mean - random_train_std, random_train_mean + random_train_std, alpha=0.2, color="red")

    axes[0].plot(x_axis, random_val_mean, label="Validation (RS)", color="red", linewidth=2)
    axes[0].fill_between(x_axis, random_val_mean - random_val_std, random_val_mean + random_val_std, alpha=0.2, color="red")

    axes[0].axhline(y=args.supervised_accuracy, color='black', linestyle='dashed', linewidth=1.5, label='Supervised Baseline')

    axes[0].set_xlabel("Number of Labeled Data Points", fontsize=12)
    axes[0].set_ylabel("Accuracy", fontsize=12)
    axes[0].set_title("Train & Validation Accuracy (ALS vs RS)", fontsize=14)
    axes[0].legend(fontsize=10)
    axes[0].grid(True)



    # ---- Subplot 2: Time Plot ----
    axes[1].plot(x_axis, original_times_mean, label="Original Re-training", color="red", linewidth=2)
    axes[1].fill_between(x_axis, original_times_mean - original_times_std, original_times_mean + original_times_std, alpha=0.2, color="red")

    axes[1].plot(x_axis, modified_times_mean, label="Modified Updating", color="blue", linewidth=2)
    axes[1].fill_between(x_axis, modified_times_mean - modified_times_std, modified_times_mean + modified_times_std, alpha=0.2, color="blue")

    axes[1].set_xlabel("Number of Labeled Data Points", fontsize=12)
    axes[1].set_ylabel("Time (seconds)", fontsize=12)
    axes[1].set_title("Time Scaling: Original vs Modified Updating", fontsize=14)
    axes[1].legend(fontsize=10)
    axes[1].grid(True)


    # Save and show plots
    plt.tight_layout()
    plt.savefig(os.path.join(args.logs_path, "comparison_plots.png"), dpi=300)
    plt.show()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sr_no", type=int, required=True)
    parser.add_argument("--logs_path", type=str, default="logs")
    parser.add_argument("--supervised_accuracy", type=float, required=True)
    main(parser.parse_args())