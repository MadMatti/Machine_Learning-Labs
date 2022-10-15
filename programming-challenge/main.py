

TRAIN_FILE = "programming-challenge/resources/TrainOnMe-4.csv"
TEST_FILE = "programming-challenge/resources/TestOnMe-4.csv"

if __name__ == "__main__":
    df = data.load_file(TRAIN_FILE)
    modelling.test(df)