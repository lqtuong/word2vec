import gensim, logging
import smart_open, os
import nltk
import multiprocessing
import string

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

source_dir = '/home/tuong/Downloads/Text_Classification/'

def get_filepaths(directory):
    """
    Load data file paths
    :param directory:
    :return:
    """
    file_paths = []
    for root, directories, files in os.walk(directory):
        for filename in files:
            file_path = os.path.join(root, filename)
            file_paths.append(file_path)
    return file_paths

def preprocess():
    filenames =  get_filepaths(source_dir+'data')
    vocab = dict()
    with open(source_dir + 'metadata/data.vn.txt', 'w') as f:
        for file in filenames:
            with open(file, 'r') as fr:
                for sent in nltk.sent_tokenize(fr.read()):
                    if sent != '\n':
                        f.write(sent+'\n')
                        for w in nltk.word_tokenize(sent):
                            if w not in string.punctuation:
                                if w not in vocab:
                                    vocab[w.lower()]=1
                                else: vocab[w.lower()]+=1
    with open(source_dir+'metadata/vocab.txt', 'w') as f:
        for w in vocab:
            f.write(w+' '+str(vocab[w])+'\n')

def train():
    # sentences = gensim.models.word2vec.LineSentence(source_dir+'metadata/data.vn.txt')
    sentences = gensim.models.word2vec.Text8Corpus(source_dir+'metadata/data.vn.txt')
    model = gensim.models.Word2Vec(sentences, size=2, workers=multiprocessing.cpu_count(), min_count=1)
    model.save(source_dir+'word2vec/word2vec.vn.bin')
    print(model.wv.vocab)

def load(dir):
    model = gensim.models.Word2Vec.load(dir)
    vector = {}
    for word in model.wv.vocab.keys():
        vector[word] = model[word]

    print(vector)

if __name__=='__main__':
    #preprocess()
    #train()
    load(source_dir+'word2vec/word2vec.vn.bin')