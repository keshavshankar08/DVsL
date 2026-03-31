from dvsl_backend import *

# define class
dvsl = dvsl_backend()

# read in words from files bad_words.txt and good_words.txt
with open("good_words.txt", "r") as f:
    good_words = [line.strip() for line in f.readlines()]
    
with open("bad_words.txt", "r") as f:
    bad_words = [line.strip() for line in f.readlines()]

# ensure same size between lists of words in files
assert len(good_words) == len(bad_words), "The files do not have the same number of words!"

# loop through words
count = 0
for good_word, bad_word in zip(good_words, bad_words):
    # use llm to correct bad word
    good_word_prediction = dvsl.llm_correct(bad_word)

    print(f"True: {good_word}\tBad: {bad_word}\tCorrected: {good_word_prediction}")

    # check if matches true good word
    if good_word == good_word_prediction:
        count += 1

# calculate final accuracy of llm spell checker
accuracy = count / len(good_words)
print(f"Accuracy: {accuracy}")