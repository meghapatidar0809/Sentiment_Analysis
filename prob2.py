import os
import numpy as np
import pickle
import argparse
import time
from scipy.stats import entropy

from utils import *
from model import *

# Using Gini Impurity
def calculate_uncertainty(model, X):    
    probability = model.predict_proba(X)                # Get the predicted probabilities for each class
    gini_impurity = 1 - np.sum(probability**2, axis=1)  # Calculate Gini Impurity for each data point
    return gini_impurity

# Using entropy
def calculate_uncertainty_entropy(model, X):
    probability = model.predict_proba(X)         # Get the predicted probabilities for each class    
    uncertainties = entropy(probability, axis=1) # Calculate entropy for each data point
    return uncertainties

def main(args: argparse.Namespace):
    # Set seed for reproducibility
    assert args.run_id is not None and 0 < args.run_id < 6, "Invalid run_id"
    set_seed(args.sr_no + args.run_id)

    # Load the preprocessed data
    if os.path.exists(f"{args.data_path}/X_train{args.intermediate}"):
        X_train_vec = pickle.load(open(f"{args.data_path}/X_train{args.intermediate}", "rb"))
        X_val_vec = pickle.load(open(f"{args.data_path}/X_val{args.intermediate}", "rb"))
        y_train = pickle.load(open(f"{args.data_path}/y_train{args.intermediate}", "rb"))
        y_val = pickle.load(open(f"{args.data_path}/y_val{args.intermediate}", "rb"))
        idxs = np.random.RandomState(args.run_id).permutation(X_train_vec.shape[0])
        X_train_vec = X_train_vec[idxs]
        y_train = y_train[idxs]
        print("Preprocessed Data Loaded")
    else:
        raise Exception("Preprocessed Data not found")

    # Train the model
    model = MultinomialNaiveBayes(alpha=args.smoothing)
    accs = []
    total_items = 10_000
    idxs = np.arange(10_000)
    remaining_idxs = np.setdiff1d(np.arange(X_train_vec.shape[0]), idxs)
    
    # Lists to store timing data
    original_times = []
    modified_times = []

    train_accs = []  # List to store training accuracy

    # Train the model
    for i in range(1, 60):
        X_train_batch = X_train_vec[idxs]
        y_train_batch = y_train[idxs]
    
        if i == 1:
            start_time = time.time()
            model.fit(X_train_batch, y_train_batch)
            original_times.append(time.time() - start_time)
        else:
            start_time = time.time()
            model.fit(X_train_batch, y_train_batch, update=True)
            original_times.append(time.time() - start_time)
        
        # Time the method
        start_time = time.time()
        model.fit(X_train_batch, y_train_batch, update=True)  # Update incrementally
        modified_times.append(time.time() - start_time)
        
        # Compute validation accuracy
        y_preds_val = model.predict(X_val_vec)
        val_acc = np.mean(y_preds_val == y_val)
        print(f"{total_items} items - Val acc: {val_acc}")
        accs.append(val_acc)
        
        # Compute training accuracy
        y_preds_train = model.predict(X_train_batch)
        train_acc = np.mean(y_preds_train == y_train_batch)
        train_accs.append(train_acc)

        if args.is_active:            
            # Calculate uncertainty for remaining unlabeled data
            uncertainties = calculate_uncertainty(model, X_train_vec[remaining_idxs])
            
            # Select top 5k uncertain points
            selected_idxs = remaining_idxs[np.argsort(uncertainties)[-10_000:]]
            
            # Update labeled & unlabeled 
            idxs = selected_idxs          # Only use the newly selected points
            remaining_idxs = np.setdiff1d(remaining_idxs, selected_idxs)
        else:
            idxs = remaining_idxs[:10_000] # Only use the newly selected points
            remaining_idxs = remaining_idxs[10_000:]
            
        total_items += 10_000
     
    # Save training accuracy
    train_accuracy = np.array(train_accs)
    np.save(f"{args.logs_path}/train_run_{args.run_id}_{args.is_active}.npy", train_accuracy)
        
    # Save validation accuracy
    val_accuracy = np.array(accs)
    np.save(f"{args.logs_path}/run_{args.run_id}_{args.is_active}.npy", val_accuracy)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sr_no", type=int, required=True)
    parser.add_argument("--run_id", type=int, required=True)
    parser.add_argument("--is_active", action="store_true")
    parser.add_argument("--data_path", type=str, default="data")
    parser.add_argument("--logs_path", type=str, default="logs")
    parser.add_argument("--intermediate", type=str, default="_i.pkl")
    parser.add_argument("--max_vocab_len", type=int, default=10_000)
    parser.add_argument("--smoothing", type=float, default=0.1)
    main(parser.parse_args())