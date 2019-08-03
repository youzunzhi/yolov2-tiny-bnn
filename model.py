import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import tqdm
import os, time

from modules import *
from utils.computation import *
from utils.utils import log_train_progress, show_eval_result
from utils.dataset import get_imgs_size

class BaseModel(object):
    def set_train_state(self, *names):
        """
        set the given attributes in names to the training state.
        if names is empty, call the train() method for all attributes which are instances of nn.Module.
        :param names:
        :return:
        """
        if not names:
            modules = []
            for attr_name in dir(self):
                attr = getattr(self, attr_name)
                if isinstance(attr, nn.Module):
                    modules.append(attr_name)
        else:
            modules = names

        for m in modules:
            getattr(self, m).train()

    def set_eval_state(self, *names):
        """
        set the given attributes in names to the evaluation state.
        if names is empty, call the eval() method for all attributes which are instances of nn.Module.
        :param names:
        :return:
        """
        if not names:
            modules = []
            for attr_name in dir(self):
                attr = getattr(self, attr_name)
                if isinstance(attr, nn.Module):
                    modules.append(attr_name)
        else:
            modules = names

        for m in modules:
            getattr(self, m).eval()

class Model(BaseModel):
    def __init__(self, options, logger, training=True):
        super(BaseModel, self).__init__()
        self.options = options
        self.logger = logger
        self.modules_defs = self.parse_model_cfg(options.model_cfg)
        self.hyper_parameters, self.modules = self.get_modules()
        self.batch_size = self.options.batch_size
        self.training = training
        if training:
            self.seen = 0
            self.header_info = np.array([0, 0, 0, self.seen], dtype=np.int32)
            self.save_weights_fname = options.model_cfg.split('/')[1].split('.')[0] + logger.time_string() + '.weights'
            self.trained = self.options.trained
            self.no_pretrained = self.options.no_pretrained

            if self.trained:
                self.learning_rate = float(self.hyper_parameters['learning_rate']) * 0.01
                weights_file = self.options.weights_file
            elif self.no_pretrained:
                self.learning_rate = float(self.hyper_parameters['learning_rate'])
                weights_file = 'no pretrain'
            else:
                self.learning_rate = float(self.hyper_parameters['learning_rate']) * 0.01
                weights_file = self.options.pretrained_weights

            decay = float(self.hyper_parameters['decay'])
            self.optimizer = optim.SGD(self.modules_list.parameters(),
                                       lr=self.learning_rate/self.batch_size,
                                       momentum=float(self.hyper_parameters['momentum']),
                                       weight_decay=decay*self.batch_size)
        else:
            weights_file = self.options.weights_file

        self.load_weights(weights_file)

    def train(self, train_dataloader, eval_dataloader):
        total_epochs = self.options.total_epochs
        self.set_train_state()
        for epoch in range(total_epochs):
            start_time = time.time()
            for batch_i, (imgs, targets, img_path) in enumerate(train_dataloader):
                inputs = Variable(imgs.type(torch.cuda.FloatTensor))
                targets = Variable(targets.type(torch.cuda.FloatTensor), requires_grad=False)
                for module in self.modules[:-1]:
                    inputs = module(inputs)
                loss = self.modules[-1](inputs, self.seen, targets)

                self.optimizer.zero_grad()
                loss.backward()
                for p in list(self.modules_list.parameters()):
                    if hasattr(p, 'org'):
                        p.data.copy_(p.org)
                self.optimizer.step()
                for p in list(self.modules_list.parameters()):
                    if hasattr(p, 'org'):
                        p.org.copy_(p.data.clamp_(-1, 1))

                log_train_progress(epoch, total_epochs, batch_i, len(train_dataloader), self.learning_rate, start_time,
                                   self.modules_list[-1][0].metrics, self.logger)

            if epoch % self.options.eval_interval == self.options.eval_interval - 1:
                self.logger.print_log("\n---- Evaluating Model ----")
                self.eval(eval_dataloader)
            if epoch % self.options.save_interval == self.options.save_interval - 1:
                self.logger.print_log("\n---- Saving Model ----")
                fname = os.path.join(self.options.save_path, self.save_weights_fname)
                self.save_weights(fname)
            self.adjust_learning_rate(epoch)

        if total_epochs % self.options.eval_interval != self.options.eval_interval - 1:
            self.logger.print_log("\n---- Evaluating Model ----")
            self.eval(eval_dataloader)
        if total_epochs % self.options.save_interval != self.options.save_interval - 1:
            self.logger.print_log("\n---- Saving Model ----")
            fname = os.path.join(self.options.save_path, self.save_weights_fname)
            self.save_weights(fname)

    def eval(self, dataloader):
        self.set_eval_state()
        metrics = []
        labels = []
        for batch_i, (imgs, targets, imgs_path) in enumerate(tqdm.tqdm(dataloader, desc="Detecting objects")):
            labels += targets[:, 1].tolist()
            inputs = Variable(imgs.type(torch.cuda.FloatTensor), requires_grad=False)
            imgs_size = get_imgs_size(imgs_path)
            # Rescale target
            targets[:, 2:] = xywh2xyxy(targets[:, 2:])
            for target in targets:
                target[2:] *= imgs_size[target[0].long()]
            with torch.no_grad():
                for i, module in enumerate(self.modules[:-1]):
                    inputs = module(inputs)
                outputs = self.modules[-1](inputs, 0)
                outputs = outputs.cpu()

            predictions = non_max_suppression(outputs, imgs_size, self.options.conf_thresh, self.options.nms_thresh)
            metrics += get_batch_metrics(predictions, targets)

        show_eval_result(metrics, labels, self.logger)

    def parse_model_cfg(self, model_cfg_path):
        """
        Parses the yolov2-tiny layer configuration file and returns module definitions(list of dicts)
        """
        model_cfg_file = open(model_cfg_path, 'r')
        lines = model_cfg_file.read().split('\n')
        lines = [x for x in lines if x and not x.startswith('#')]  # get rid of comments
        lines = [x.rstrip().lstrip() for x in lines]  # get rid of fringe whitespaces
        module_defs = []
        for line in lines:
            if line.startswith('['):  # This marks the start of a new block
                module_defs.append({})
                module_defs[-1]['type'] = line[1:-1].rstrip()
                if module_defs[-1]['type'] == 'convolutional':
                    module_defs[-1]['batch_normalize'] = 0
            else:
                key, value = line.split("=")
                value = value.strip()
                module_defs[-1][key.rstrip()] = value.strip()

        return module_defs

    def get_modules(self):
        """
            Constructs modules(nn.Sequential) of layer blocks from module configuration in modules_defs
        """
        hyper_parameters = self.modules_defs.pop(0)
        output_filters = [int(hyper_parameters["channels"])]
        modules = nn.Sequential()
        binarize_flag = False
        for modules_i, modules_def in enumerate(self.modules_defs):

            if modules_def["type"] == "convolutional":
                bn = int(modules_def["batch_normalize"])
                filters = int(modules_def["filters"])
                kernel_size = int(modules_def["size"])
                stride = int(modules_def["stride"])
                pad = (kernel_size - 1) // 2
                binarize = int(modules_def["binarize"]) if "binarize" in modules_def else 0
                if binarize:
                    binarize_flag = True
                    modules.add_module(
                        f"bin_conv_{modules_i}",
                        BinarizeConv2d(
                            in_channels=output_filters[-1],
                            out_channels=filters,
                            kernel_size=kernel_size,
                            stride=stride,
                            padding=pad,
                            bias=not bn,
                        ),
                    )
                    module_def_pool = self.modules_defs[modules_i + 1]
                    if module_def_pool["type"] == "maxpool":
                        pool_size = int(module_def_pool['size'])
                        stride = int(module_def_pool['stride'])
                        if stride > 1:
                            maxpool = nn.MaxPool2d(pool_size, stride)
                        else:
                            maxpool = MaxPoolStride1()
                        modules.add_module(f"maxpool_{modules_i}", maxpool)
                else:
                    modules.add_module(
                        f"conv_{modules_i}",
                        nn.Conv2d(
                            in_channels=output_filters[-1],
                            out_channels=filters,
                            kernel_size=kernel_size,
                            stride=stride,
                            padding=pad,
                            bias=not bn,
                        ),
                    )
                if bn:
                    modules.add_module(
                        f"batchnorm_{modules_i}",
                        nn.BatchNorm2d(filters, momentum=0.9, eps=1e-5)
                    )
                if modules_def["activation"] == "leaky":
                    modules.add_module(
                        f"leaky_{modules_i}",
                        nn.LeakyReLU(0.1, inplace=True)
                    )
                elif modules_def["activation"] == "tanh":
                    modules.add_module(
                        f"tanh_{modules_i}",
                        nn.Hardtanh(inplace=True)
                    )
                elif modules_def["activation"] == "prelu":
                    modules.add_module(
                        f"prelu_{modules_i}",
                        nn.PReLU()
                    )
                output_filters.append(filters)
            elif modules_def["type"] == "maxpool":
                if binarize_flag:
                    binarize_flag = False
                else:
                    pool_size = int(modules_def['size'])
                    stride = int(modules_def['stride'])
                    if stride > 1:
                        maxpool = nn.MaxPool2d(pool_size, stride)
                    else:
                        maxpool = MaxPoolStride1()
                    modules.add_module(f"maxpool_{modules_i}", maxpool)

            elif modules_def["type"] == "region":
                region = RegionLoss(modules_def)
                modules.add_module(f"region_{modules_i}", region)

            else:
                print('unknown type %s' % (modules_def['type']))

        modules = modules.cuda()

        return hyper_parameters, modules

    def adjust_learning_rate(self, epoch):
        if epoch == 60:
            self.learning_rate *= 0.1
        elif epoch == 90:
            self.learning_rate *= 0.1
        else:
            return
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.learning_rate / self.batch_size

    def load_weights(self, weights_file):
        """Parses and loads the weights stored in 'weights_file'"""
        if not os.path.exists(weights_file):
            self.logger.print_log(weights_file+' does not exist, no pretrained weights loaded.')
            return
        # Open the weights file
        with open(weights_file, "rb") as f:
            header = np.fromfile(f, dtype=np.int32, count=4)  # First four are header values
            self.header_info = header  # Needed to write header when saving weights
            if self.training:
                self.seen = header[3]  # number of images seen during training
            weights = np.fromfile(f, dtype=np.float32)  # The rest are weights

        # Establish cutoff for loading backbone weights
        cutoff = None
        if "darknet.weights" in weights_file:
            cutoff = 13

        ptr = 0
        for i, module in enumerate(self.modules):
            if i == cutoff:
                break
            if isinstance(module, nn.Conv2d):
                conv_layer = module
                if not isinstance(self.modules[i+1], nn.BatchNorm2d):
                    # Load conv. bias
                    num_b = conv_layer.bias.numel()
                    conv_b = torch.from_numpy(weights[ptr: ptr + num_b]).view_as(conv_layer.bias)
                    conv_layer.bias.data.copy_(conv_b)
                    ptr += num_b
                    # Load conv. weights
                    num_w = conv_layer.weight.numel()
                    conv_w = torch.from_numpy(weights[ptr: ptr + num_w]).view_as(conv_layer.weight)
                    conv_layer.weight.data.copy_(conv_w)
                    ptr += num_w
            elif isinstance(module, nn.BatchNorm2d):
                bn_layer = module
                num_b = bn_layer.bias.numel()  # Number of biases
                # Bias
                bn_b = torch.from_numpy(weights[ptr: ptr + num_b]).view_as(bn_layer.bias)
                bn_layer.bias.data.copy_(bn_b)
                ptr += num_b
                # Weight
                bn_w = torch.from_numpy(weights[ptr: ptr + num_b]).view_as(bn_layer.weight)
                bn_layer.weight.data.copy_(bn_w)
                ptr += num_b
                # Running Mean
                bn_rm = torch.from_numpy(weights[ptr: ptr + num_b]).view_as(bn_layer.running_mean)
                bn_layer.running_mean.data.copy_(bn_rm)
                ptr += num_b
                # Running Var
                bn_rv = torch.from_numpy(weights[ptr: ptr + num_b]).view_as(bn_layer.running_var)
                bn_layer.running_var.data.copy_(bn_rv)
                ptr += num_b
                # Load conv. weights
                num_w = conv_layer.weight.numel()
                conv_w = torch.from_numpy(weights[ptr: ptr + num_w]).view_as(conv_layer.weight)
                conv_layer.weight.data.copy_(conv_w)
                ptr += num_w

    def save_weights(self, fname, cutoff=-1):
        """
        :param path: path of the new weights file
        :param cutoff: save layers between 0 and cutoff (cutoff == -1 -> all save)
        :return:
        """
        fp = open(fname, "wb")
        self.header_info[3] = self.seen
        self.header_info.tofile(fp)

        # Iterate through layers
        for i, modules in enumerate(self.modules[:cutoff]):
            if isinstance(module, nn.Conv2d):
                conv_layer = module
                if not isinstance(self.modules[i + 1], nn.BatchNorm2d):
                    conv_layer.bias.data.cpu().numpy().tofile(fp)
                    conv_layer.weight.data.cpu().numpy().tofile(fp)

            elif isinstance(module, nn.BatchNorm2d):
                bn_layer = module
                bn_layer.bias.data.cpu().numpy().tofile(fp)
                bn_layer.weight.data.cpu().numpy().tofile(fp)
                bn_layer.running_mean.data.cpu().numpy().tofile(fp)
                bn_layer.running_var.data.cpu().numpy().tofile(fp)
                conv_layer.weight.data.cpu().numpy().tofile(fp)

        fp.close()

        self.logger.print_log('Saved weights to '+fname)