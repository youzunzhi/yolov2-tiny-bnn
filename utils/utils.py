import argparse
import os, sys, time
import torch


class Options():
    def __init__(self):
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        # training options
        parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
        parser.add_argument("--batch_size", type=int, default=8, help="size of each image batch")
        parser.add_argument("--multiscale_training", default=True, help="allow for multi-scale training")
        # data and model
        parser.add_argument("--data_cfg", type=str, default="cfg/voc.data", help="path to data cfg file")
        parser.add_argument("--model_cfg", type=str, default="cfg/yolov2-tiny-voc.cfg",
                                 help="path to model cfg file")
        parser.add_argument("--weights_path", type=str, default="weights/yolov2-tiny-voc.weights",
                                 help="path to weights file")
        # other configs
        parser.add_argument('--log_path', type=str, default='./logs/', help='Folder to save checkpoints and log.')
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
        ISOTIMEFORMAT = '%Y-%m-%d %X'
        string = '[{}]'.format(time.strftime(ISOTIMEFORMAT, time.localtime(time.time())))
        return string