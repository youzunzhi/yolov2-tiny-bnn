from utils.utils import Options, Logger
from utils.dataset import Yolov2Dataset
from model import Model
import pydevd_pycharm


if __name__ == "__main__":
    options = Options(training=True).options
    logger = Logger(options.log_path)
    logger.print_options(options)


    if options.debug:
        pydevd_pycharm.settrace('172.26.3.54', port=12344, stdoutToServer=True, stderrToServer=True)

    train_dataset = PretrainDataset(options, training=True)
    eval_dataset = PretrainDataset(options, training=False)
    train_dataloader = train_dataset.get_dataloader()
    eval_dataloader = eval_dataset.get_dataloader()

    model = Model(options, logger)
    model.pretrain(train_dataloader, eval_dataloader)

