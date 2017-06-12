import argparse
from data import PTBIterator
from train import validation
import torch


def test(data_path, model_path, dict_path, batch_size, maxlen):
    test_iter = PTBIterator(data_path=data_path, dict=dict_path, batch_size=batch_size, maxlen=maxlen)
    with open(model_path, 'rb') as f:
        HM_model = torch.load(f)
    PPL = validation(test_iter, HM_model)
    print 'test perplexity: ', PPL

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-data', type=str, default="../data/PTB/test.txt", help="data path")
    parser.add_argument('-model', type=str, default="model.pt_iter2000", help="model path")
    args = parser.parse_args()

    test(data_path=args.data, model_path=args.model, dict_path='../data/PTB/dict.pkl',
         batch_size=80, maxlen=35)
