import argparse
import os, sys, time
import torch
import numpy as np
from terminaltables import AsciiTable
import time
import datetime

from utils.computation import ap_per_class


class Options():
    def __init__(self, training):
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        if training:
            # training options
            parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
            parser.add_argument("--multiscale_training", default=True, help="allow for multi-scale training")

        # data and model
        parser.add_argument("--data_cfg", type=str, default="cfg/voc.data", help="path to data cfg file")
        parser.add_argument("--model_cfg", type=str, default="cfg/yolov2-tiny-voc.cfg",
                                 help="path to model cfg file")
        parser.add_argument("--weights_file", type=str, default="weights/yolov2-tiny-voc.weights",
                                 help="path to weights file")
        # hyper parameters
        parser.add_argument("--batch_size", type=int, default=8, help="size of each image batch")
        parser.add_argument("--conf_thresh", type=float, default=0.25, help="only keep detections with conf higher than conf_thresh")
        parser.add_argument("--nms_thresh", type=float, default=0.4, help="the threshold of non-max suppresion algorithm")
        # other configs
        parser.add_argument('--log_path', type=str, default='./logs/', help='Folder to save checkpoints and log.')
        parser.add_argument("--eval_interval", type=int, default=1, help="interval of evaluations on validation set")
        parser.add_argument("--save_interval", type=int, default=10, help="interval of saving model weights")
        parser.add_argument('--save_path', type=str, default='./weights/', help='Folder to save checkpoints and log.')
        parser.add_argument('--gpu', type=str, default='2', help='gpu id.')
        parser.add_argument("--n_cpu", type=int, default=8,
                                 help="number of cpu threads to use during batch generation")
        parser.add_argument("--use_cuda", action='store_false', default=True,
                                 help="use cuda device or not")
        parser.add_argument("--debug", action='store_true', default=False,
                                 help="use remote debugger, make sure remote debugger is running")

        self.options = parser.parse_args()
        os.environ["CUDA_VISIBLE_DEVICES"] = self.options.gpu


class Logger(object):
    def __init__(self, save_path):
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        self.file = open(os.path.join(save_path, 'log_{}.txt'.format(self.time_string())), 'w')
        self.print_log("python version : {}".format(sys.version.replace('\n', ' ')))
        self.print_log("torch  version : {}".format(torch.__version__))

    def print_options(self, options):
        self.print_log("")
        self.print_log("----- options -----".center(120, '-'))
        options = vars(options)
        string = ''
        for i, (k, v) in enumerate(sorted(options.items())):
            string += "{}: {}".format(k, v).center(40, ' ')
            if i % 3 == 2 or i == len(options.items()) - 1:
                self.print_log(string)
                string = ''
        self.print_log("".center(120, '-'))
        self.print_log("")

    def print_log(self, string, write_file=True):
        if write_file:
            self.file.write("{}\n".format(string))
            self.file.flush()
        print(string)

    def time_string(self):
        ISOTIMEFORMAT = '%Y-%m-%d-%X'
        string = '[{}]'.format(time.strftime(ISOTIMEFORMAT, time.localtime(time.time())))
        return string

def log_train_progress(epoch, total_epochs, batch_i, total_batch, start_time, metrics, logger):
    log_str = "\n---- [Epoch %d/%d, Batch %d/%d] ----\n" % (epoch, total_epochs, batch_i, total_batch)
    metric_table = [["Metrics", "Region Layer"]]
    formats = {m: "%.6f" for m in metrics}
    formats["grid_size"] = "%2d"
    formats["cls_acc"] = "%.2f%%"
    for i, metric in enumerate(metrics):
        row_metrics = formats[metric] % metrics.get(metric, 0)
        metric_table += [[metric, row_metrics]]
    log_str += AsciiTable(metric_table).table

    # Determine approximate time left for epoch
    epoch_batches_left = total_batch - (batch_i + 1)
    time_left = datetime.timedelta(seconds=epoch_batches_left * (time.time() - start_time) / (batch_i + 1))
    log_str += f"\n---- ETA {time_left}"
    logger.print_log(log_str, write_file=True)

def show_eval_result(metrics, labels, logger):
    true_positives, pred_conf, pred_labels = [np.concatenate(x, 0) for x in list(zip(*metrics))]
    precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_conf, pred_labels, labels)
    logger.print_log(f"mAP: {AP.mean()}")
