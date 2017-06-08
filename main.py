import argparse
from train import train

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-lr', type=float, default=0.1, help="learning rate")
    parser.add_argument('-model', type=str, default="model.pt", help="save to path")
    parser.add_argument('-init', type=str, default="model.pt", help="reload path")
    parser.add_argument('-reload', action='store_true', default=False, help="reload flag")
    parser.add_argument('-batch', type=int, default=80, help="batch size")
    args = parser.parse_args()

    train(data_path=["../data/train.tok", "../data/valid.tok"], dict_path="../data/dict.pkl",
          size_list=[512, 512], dict_size=14000, embed_size=128,
          batch_size=args.batch, maxlen=100, learning_rate=args.lr, max_epoch=100,
          valid_iter=2000, show_iter=1000, init='', reload_=True, saveto='model.pt')
