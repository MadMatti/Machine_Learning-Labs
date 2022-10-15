from data import load_file, cleaning, analysis
from modelling import split, transform, test_classifiers


TRAIN_FILE = "programming-challenge/resources/TrainOnMe-4.csv"
TEST_FILE = "programming-challenge/resources/TestOnMe-4.csv"

if __name__ == "__main__":
    df = load_file(TRAIN_FILE)
    df_clean = cleaning(df)

    test_classifiers(transform(split(df_clean)), split(df_clean))
