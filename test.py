import argparse
from data import Corpus
from train import evaluatePTB
import torch
from utils import batchify
import cPickle as pkl


def test(data_path, model_path, options_path, dict_path):

    with open(model_path, 'rb') as f:
        model = torch.load(f)

    with open(options_path, 'rb') as f:
        model_params = pkl.load(f)

    print "Load data..."
    corpus = Corpus(data_path)
    test_data = batchify(corpus.test, model_params['batch_size'])
    print "Done"

    PPL = evaluatePTB(test_data, model, model_params, corpus.dictionary)
    print 'test perplexity: ', PPL

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-data', type=str, default="../data/PTB", help="data path")
    parser.add_argument('-model', type=str, default="model-1/model.pt_epoch39", help="model path")
    args = parser.parse_args()

    test(data_path=args.data, model_path=args.model, options_path='model.options.pkl', dict_path='../data/PTB/dict.pkl')
