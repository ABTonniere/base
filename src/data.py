DATASET_PATH = "./spam.csv"

# Tested multiple value 0.2 seems good
TEST_SIZE = 0.2

def get_dataframe_cleaned():

    df = pd.read_csv(DATASET_PATH, encoding="ISO-8859-1")
    df = df.drop_duplicates(subset=["message"])
    df["message_clean"] = df["message"].str.lower()
    df["message_clean"] = df["message_clean"].apply(lambda x: re.sub(r"http\S+|www\S+|https\S+", "", x))
    df["message_clean"] = df["message_clean"].apply(lambda x: re.sub(r"\S+@\S+", "", x))
    df["message_clean"] = df["message_clean"].apply(lambda x: re.sub(r"\d+", "", x))
    df["message_clean"] = df["message_clean"].apply(lambda x: x.translate(str.maketrans("", "", string.punctuation)))
    df["message_clean"] = df["message_clean"].apply(lambda x: " ".join(x.split()))
    df["message_length"] = df["message"].apply(len)
    df["word_count"] = df["message"].apply(lambda x: len(x.split()))
    df["avg_word_length"] = df["message_length"] / df["word_count"]
    df["caps_count"] = df["message"].apply(lambda x: sum(1 for c in x if c.isupper()))
    df["caps_ratio"] = df["caps_count"] / df["message_length"]
    df["special_chars"] = df["message"].apply(lambda x: sum(1 for c in x if c in "!?$€£%"))

    return df