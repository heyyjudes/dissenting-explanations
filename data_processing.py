# Script for generating the train and test datasets from the original dataset
import os
import re
import pandas as pd

if __name__ == "__main__":
    train_dataset_dict = {"text": [], "label": [], "sentiment": []}
    test_dataset_dict = {"text": [], "label": [], "sentiment": []}

    for path, subdirs, files in os.walk("data/op_spam_v1.4/"):
        for name in files:
            if os.path.join(path, name).endswith(".txt"):
                with open(os.path.join(path, name), "r") as file:
                    review = file.read().replace("\n", "")

                    if name.startswith("d_"):
                        label = 0  # "deceptive"
                    else:
                        label = 1  # "real"

                    if re.compile(r"positive").search(path):
                        sentiment = 1
                    else:
                        sentiment = -1

                    if re.compile(r"fold5").search(path):
                        test_dataset_dict["text"].append(review)
                        test_dataset_dict["label"].append(label)
                        test_dataset_dict["sentiment"].append(sentiment)
                    else:
                        train_dataset_dict["text"].append(review)
                        train_dataset_dict["label"].append(label)
                        train_dataset_dict["sentiment"].append(sentiment)

    train_df = pd.DataFrame(train_dataset_dict)
    test_df = pd.DataFrame(test_dataset_dict)

    train_df.to_csv("data/reviews_train.csv")
    test_df.to_csv("data/reviews_test.csv")