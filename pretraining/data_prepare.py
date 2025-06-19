from datasets import load_dataset
import os

os.makedirs("data", exist_ok=True)

# 2a) PG-19
pg19 = load_dataset("pg19", split=["train", "validation"])
# 2b) Books3 (from The Pile)
books3 = load_dataset("the_pile", "books3", split="train")

def write_split(split, fname, min_chars=1000):
    with open(fname, "w", encoding="utf-8") as f:
        for ex in split:
            txt = ex["text"].strip().replace("\n", " ")
            if len(txt) >= min_chars:
                f.write(txt + "\n")

# 3) Write out
write_split(pg19["train"],      "data/train_pg19.txt")
write_split(pg19["validation"], "data/val_pg19.txt")
# write_split(books3,             "data/train_books3.txt")

# 4) Combine
with open("data/train.txt", "w", encoding="utf-8") as f:
    with open("data/train_pg19.txt", "r", encoding="utf-8") as pg19_file:
        f.write(pg19_file.read())
    # with open("data/train_books3.txt", "r", encoding="utf-8") as books3_file:
    #     f.write(books3_file.read())

# 5) Remove individual files
os.remove("data/train_pg19.txt")
# os.remove("data/train_books3.txt")

# 6) Final validation split
with open("data/val.txt", "w", encoding="utf-8") as f:
    with open("data/val_pg19.txt", "r", encoding="utf-8") as pg19_file:
        f.write(pg19_file.read())