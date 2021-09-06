import tensorflow as tf
import zipfile
import collections

data_index = 0

def read_data(filename):
    with zipfile.ZipFile(filename) as f:
        # Convert text to list word
        data = tf.compat.as_str(f.read(f.namelist()[0])).split()
    return data

def build_dataset(words, n_words=100000):
    count = [['UNK', -1]]
    # bag of word : count
    count.extend(collections.Counter(words.split(' ')).most_common(n_words-1))
    dictionary = dict()
    # create dict of word and count index
    for word, _ in count:
        #print(word,": ", len(dictionary))
        dictionary[word] = len(dictionary)
    # data: convert word to index, index of unk = 0, count unk_count
    data = list()
    unk_count = 0
    for word in words.split(' '):
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    # save dict: {index : word}
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    print(data)
    with open("w2w_dataset.txt", 'w') as out_file:
        out_file.write(str({"data": data, "count": count , "dictionary": dictionary, "reversed_dictionary": reversed_dictionary}))

if __name__ == "__main__":
    #data = read_data('data.zip')
    with open("data.txt",'r') as f:
        build_dataset(f.read())
