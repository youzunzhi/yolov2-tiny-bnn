from utils.utils import Options, Logger
from torch.utils.data import DataLoader
from utils.dataset import Yolov2Dataset
from model import Model
import pydevd_pycharm

if __name__ == "__main__":
    options = Options(training=False).options
    logger = Logger(options.log_path)
    logger.print_options(options)
    if options.debug:
        pydevd_pycharm.settrace('172.26.3.54', port=12344, stdoutToServer=True, stderrToServer=True)

    dataset = Yolov2Dataset(options, training=False)
    dataloader = DataLoader(
        dataset,
        batch_size=options.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=dataset.collate_fn
    )

    model = Model(options, logger, training=False)
    model.eval(dataloader)
