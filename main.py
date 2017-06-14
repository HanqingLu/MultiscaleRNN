import argparse
from train import train

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-lr', type=float, default=20.0, help="learning rate")
    parser.add_argument('-model', type=str, default="model.pt", help="save to path")
    parser.add_argument('-init', type=str, default="model.pt", help="reload path")
    parser.add_argument('-reload', action='store_true', default=False, help="reload flag")
    parser.add_argument('-batch', type=int, default=20, help="batch size")
    parser.add_argument('-clip', type=float, default=0.25, help="gradient clip")
    args = parser.parse_args()

    train(data_path=["../data/train.tok", "../data/valid.tok"], dict_path="../data/dict.pkl",
          size_list=[650, 650], dict_size=10000, embed_size=650, batch_size=args.batch, maxlen=100,
          learning_rate=args.lr, clip=args.clip, max_epoch=40,
          valid_iter=500, show_iter=500, init=args.init, reload_=args.reload, saveto='model-1/model.pt')
