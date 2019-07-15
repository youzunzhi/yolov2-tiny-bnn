from utils.utils import Options, Logger
from torch.utils.data import DataLoader
from utils.dataset import Yolov2Dataset
from model import Model
import pydevd_pycharm

if __name__ == "__main__":
    options = Options().options
    logger = Logger(options.log_path)
    logger.print_options(options)

    if options.debug:
        pydevd_pycharm.settrace('172.26.3.54', port=12344, stdoutToServer=True, stderrToServer=True)

    train_dataset = Yolov2Dataset(options, training=True, multiscale=options.multiscale)
    eval_dataset = Yolov2Dataset(options, training=False)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=options.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=train_dataset.collate_fn
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=options.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=eval_dataset.collate_fn
    )

    model = Model(options, logger)
    model.load_weights(options.weights_path)
    model.train(options, train_dataloader, eval_dataloader)
