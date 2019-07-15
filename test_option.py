import argparse
import os

class Options():
    def __init__(self):
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        # data and model
        parser.add_argument("--data_cfg", type=str, default="config/voc.data", help="path to data cfg file")
        parser.add_argument("--model_cfg", type=str, default="config/yolov2-tiny-voc.cfg",
                                 help="path to model cfg file")
        parser.add_argument("--weights_path", type=str, default="weights/yolov2-tiny-voc.weights",
                                 help="path to weights file")
        # other configs
        parser.add_argument('--gpu', type=str, default='2', help='gpu id.')
        parser.add_argument("--n_cpu", type=int, default=8,
                                 help="number of cpu threads to use during batch generation")
        parser.add_argument("--use_cuda", action='store_false', default=True,
                                 help="use cuda device or not")
        parser.add_argument("--debug", action='store_true', default=False,
                                 help="use remote debugger, make sure remote debugger is running")

        self.options = parser.parse_args()
        os.environ["CUDA_VISIBLE_DEVICES"] = self.options.gpu
        self.print_options()

    def print_options(self):
        pass
