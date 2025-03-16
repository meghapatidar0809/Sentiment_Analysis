import os

from utils import *
from model import *


def main(args: argparse.Namespace):
    # Set seed for reproducibility
    set_seed(args.sr_no)

    # Load the data
    X_train, y_train, X_val, y_val = get_data(
        path=os.path.join(args.data_path, args.train_file), seed=args.sr_no)
    print("Data Loaded")
    
    # Preprocess the data
    vectorizer = Vectorizer(max_vocab_len=args.max_vocab_len)
    vectorizer.fit(X_train)
    if os.path.exists(f"{args.data_path}/X_train{args.intermediate}"):
        X_train_vec = pickle.load(open(
            f"{args.data_path}/X_train{args.intermediate}", "rb"))
        y_train = pickle.load(open(
            f"{args.data_path}/y_train{args.intermediate}", "rb"))
        X_val_vec = pickle.load(open(
            f"{args.data_path}/X_val{args.intermediate}", "rb"))
        y_val = pickle.load(open(
            f"{args.data_path}/y_val{args.intermediate}", "rb"))
        print("Preprocessed Data Loaded")
    else:
        X_train_vec = vectorizer.transform(X=X_train)
        pickle.dump(
            X_train_vec,
            open(f"{args.data_path}/X_train{args.intermediate}", "wb"))
        pickle.dump(
            y_train, open(f"{args.data_path}/y_train{args.intermediate}", "wb"))
        X_val_vec = vectorizer.transform(X=X_val)
        pickle.dump(
            X_val_vec,
            open(f"{args.data_path}/X_val{args.intermediate}", "wb"))
        pickle.dump(
            y_val, open(f"{args.data_path}/y_val{args.intermediate}", "wb"))
        print("Data Preprocessed")


    # Train the model
    model = MultinomialNaiveBayes(alpha=args.smoothing)
    model.fit(X_train_vec, y_train)
    print("Model Trained")


    # Evaluate the model
    y_pred = model.predict(X_train_vec)
    print(f"Train Accuracy: {np.mean(y_pred == y_train)}")

    y_pred = model.predict(X_val_vec)
    print(f"Validation Accuracy: {np.mean(y_pred == y_val)}")

    # TODO: Note down the validation accuracy
    with open("Validation_accuracy_log.txt", "a") as log_file:
        log_file.write(f"Validation Accuracy: {np.mean(y_pred == y_val)}\n")


    # Load the test data
    if os.path.exists(f"{args.data_path}/X_test{args.intermediate}"):
        X_test_vec = pickle.load(open(
            f"{args.data_path}/X_test{args.intermediate}", "rb"))
        print("Preprocessed Test Data Loaded")
    else:
        X_test = pd.read_csv(
            f"{args.data_path}/X_test_{args.sr_no}.csv", header=None
        ).values.squeeze()
        print("Test Data Loaded")
        X_test_vec = vectorizer.transform(X=X_test)
        pickle.dump(
            X_test_vec,
            open(f"{args.data_path}/X_test{args.intermediate}", "wb"))
        print("Test Data Preprocessed")
    preds = model.predict(X_test_vec)
    with open(f"predictions.csv", "w") as f:
        for pred in preds:
            f.write(f"{pred}\n")
    print("Predictions Saved to predictions.csv")
    print("You may upload the file at http://10.192.30.174:8000/submit")
    
    
    # Computing negative log likelihoods
    log_probs = np.vstack([
        X_val_vec[i] @ model.sum_means.T + model.priors                  # Compute log probabilities for each sample
        for i in range(X_val_vec.shape[0])
    ])

    nll = -np.mean([log_probs[i, y_val[i]] for i in range(len(y_val))])  # Compute NLL
    print(f"Validation Negative Log-Likelihood: {nll}")


    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sr_no", type=int, required=True, help="Student roll number")
    parser.add_argument("--data_path", type=str, default="data", help="Path to data files")
    parser.add_argument("--train_file", type=str, default="train.csv", help="Training data file")
    parser.add_argument("--intermediate", type=str, default="_i.pkl", help="Intermediate file suffix")
    parser.add_argument("--max_vocab_len", type=int, default=10_000, help="Maximum vocabulary size")
    parser.add_argument("--smoothing", type=float, default=0.1, help="Laplace smoothing parameter")
    main(parser.parse_args())