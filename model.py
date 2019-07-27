import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import tqdm
import os, time

from modules import *
from utils.computation import *
from utils.utils import OptimizerManager, parse_model_cfg, log_train_progress, show_eval_result
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

    def pretrain(self, train_dataloader, eval_dataloader):
        modules_def = parse_model_cfg(self.options.pretrain_model_cfg)
        hyper_parameters, module_list = self.get_module_list(modules_def)
        optimizer_manager = OptimizerManager(module_list, 'pretrain', self.options.batch_size)
        optimizer = optimizer_manager.get_optimizer()
        self.set_train_state()
        total_epochs = self.options.total_epochs
        for epoch in range(total_epochs):
            start_time = time.time()
            for batch_i, (imgs, targets, img_path) in enumerate(train_dataloader):
                inputs = Variable(imgs.type(torch.cuda.FloatTensor))
                targets = Variable(targets.type(torch.cuda.FloatTensor), requires_grad=False)
                for i, (module_def, module) in enumerate(zip(modules_def, module_list)):
                    if module_def["type"] in ["convolutional", "upsample", "maxpool"]:
                        inputs = module(inputs)
                    elif module_def["type"] == "softmax":
                        inputs = inputs.view(inputs.size(0), -1)
                        outputs = module[0](inputs)
                        loss = nn.BCELoss(outputs, targets)

                    else:
                        print('unknown type %s' % (module_def['type']))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                log_train_progress(epoch, total_epochs, batch_i, len(train_dataloader), optimizer.lr, start_time,
                                   {'loss': loss.item()}, self.logger)

                optimizer_manager.adjust_learning_rate(epoch)

            if epoch % self.options.eval_interval == self.options.eval_interval - 1:
                self.logger.print_log("\n---- Evaluating Model ----")
                self.eval(eval_dataloader, modules_def, module_list)
            if epoch % self.options.save_interval == self.options.save_interval - 1:
                self.logger.print_log("\n---- Saving Model ----")
                save_weights_fname = self.logger.time_string() + '_' + str(epoch)
                fname = os.path.join(self.options.save_path, save_weights_fname)
                self.save_weights(fname)

        if total_epochs % self.options.eval_interval != self.options.eval_interval - 1:
            self.logger.print_log("\n---- Evaluating Model ----")
            self.eval(eval_dataloader, modules_def, module_list)
        if total_epochs % self.options.save_interval != self.options.save_interval - 1:
            self.logger.print_log("\n---- Saving Model ----")
            save_weights_fname = self.logger.time_string() + '_' + str(total_epochs)
            fname = os.path.join(self.options.save_path, save_weights_fname)
            self.save_weights(fname)


    def train(self, train_dataloader, eval_dataloader):
        modules_def = parse_model_cfg(self.options.model_cfg)
        hyper_parameters, module_list = self.get_module_list(modules_def)
        self.load_weights(self.options.weights_file, modules_def, module_list)
        optimizer_manager = OptimizerManager(module_list, 'train', self.options.batch_size)
        optimizer = optimizer_manager.get_optimizer()
        self.set_train_state()

        total_epochs = self.options.total_epochs
        for epoch in range(total_epochs):
            start_time = time.time()
            for batch_i, (imgs, targets, img_path) in enumerate(train_dataloader):
                inputs = Variable(imgs.type(torch.cuda.FloatTensor))
                targets = Variable(targets.type(torch.cuda.FloatTensor), requires_grad=False)
                for i, (module_def, module) in enumerate(zip(modules_def, module_list)):
                    if module_def["type"] in ["convolutional", "maxpool", "avgpool"]:
                        inputs = module(inputs)
                    elif module_def["type"] == "region":
                        loss = module[0](inputs, self.seen, targets)
                        self.seen += inputs.shape[0]
                    else:
                        print('unknown type %s' % (module_def['type']))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                log_train_progress(epoch, total_epochs, batch_i, len(train_dataloader), optimizer.lr, start_time,
                                   module_list[-1][0].metrics, self.logger)

                optimizer_manager.adjust_learning_rate(epoch)

            if epoch % self.options.eval_interval == self.options.eval_interval - 1:
                self.logger.print_log("\n---- Evaluating Model ----")
                self.eval(eval_dataloader, modules_def, module_list)
            if epoch % self.options.save_interval == self.options.save_interval - 1:
                self.logger.print_log("\n---- Saving Model ----")
                save_weights_fname = self.logger.time_string() + '_' + str(epoch)
                fname = os.path.join(self.options.save_path, save_weights_fname)
                self.save_weights(fname)

        if total_epochs % self.options.eval_interval != self.options.eval_interval - 1:
            self.logger.print_log("\n---- Evaluating Model ----")
            self.eval(eval_dataloader, modules_def, module_list)
        if total_epochs % self.options.save_interval != self.options.save_interval - 1:
            self.logger.print_log("\n---- Saving Model ----")
            save_weights_fname = self.logger.time_string() + '_' + str(total_epochs)
            fname = os.path.join(self.options.save_path, save_weights_fname)
            self.save_weights(fname)

    def eval(self, dataloader, modules_def=None, module_list=None):
        if modules_def is None or module_list is None:
            modules_def = parse_model_cfg(self.options.model_cfg)
            hyper_parameters, module_list = self.get_module_list(modules_def)

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
                for i, (module_def, module) in enumerate(zip(modules_def, module_list)):
                    if module_def["type"] in ["convolutional", "upsample", "maxpool"]:
                        inputs = module(inputs)
                    elif module_def["type"] == "region":
                        outputs = module[0](inputs, 0)
                        outputs = outputs.cpu()
                    else:
                        print('unknown type %s' % (module_def['type']))

            predictions = non_max_suppression(outputs, imgs_size, self.options.conf_thresh, self.options.nms_thresh)
            metrics += get_batch_metrics(predictions, targets)

        show_eval_result(metrics, labels, self.logger)

    def get_module_list(self, modules_def):
        """
            Constructs module list of layer blocks from module configuration in modules_def
        """
        hyper_parameters = modules_def.pop(0)
        output_filters = [int(hyper_parameters["channels"])]
        module_list = nn.ModuleList()
        for module_i, module_def in enumerate(modules_def):
            modules = nn.Sequential()

            if module_def["type"] == "convolutional":
                bn = int(module_def["batch_normalize"])
                filters = int(module_def["filters"])
                kernel_size = int(module_def["size"])
                stride = int(module_def["stride"])
                pad = (kernel_size - 1) // 2
                binarize = int(module_def["binarize"]) if "binarize" in module_def else 0
                if binarize:
                    modules.add_module(
                        f"bin_conv_{module_i}",
                        BinarizeConv2d(
                            in_channels=output_filters[-1],
                            out_channels=filters,
                            kernel_size=kernel_size,
                            stride=stride,
                            padding=pad,
                            bias=not bn,
                        ),
                    )
                else:
                    modules.add_module(
                        f"conv_{module_i}",
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
                        f"batchnorm_{module_i}",
                        nn.BatchNorm2d(filters, momentum=0.9, eps=1e-5)
                    )
                if module_def["activation"] == "leaky":
                    modules.add_module(
                        f"leaky_{module_i}",
                        nn.LeakyReLU(0.1, inplace=True)
                    )
                output_filters.append(filters)
            elif module_def["type"] == "maxpool":
                pool_size = int(module_def['size'])
                stride = int(module_def['stride'])
                if stride > 1:
                    maxpool = nn.MaxPool2d(pool_size, stride)
                else:
                    maxpool = MaxPoolStride1()
                modules.add_module(f"maxpool_{module_i}", maxpool)
            elif module_def["type"] == "avgpool":
                avgpool = nn.AdaptiveAvgPool2d((1,1))
                modules.add_module(f"avgpool{module_i}", avgpool)

            elif module_def["type"] == "region":
                region = RegionLoss(module_def)
                modules.add_module(f"region_{module_i}", region)

            elif module_def["type"] == "softmax":
                softmax = nn.Softmax()
                modules.add_module(f"softmax_{module_i}", softmax)

            else:
                print('unknown type %s' % (module_def['type']))

            modules = modules.cuda()
            module_list.append(modules)

        return hyper_parameters, module_list

    def load_weights(self, weights_file, modules_def, module_list, seen=None):
        """Parses and loads the weights stored in 'weights_file'"""
        if not os.path.exists(weights_file):
            self.logger.print_log(weights_file+' does not exist, no pretrained weights loaded.')
            return
        # Open the weights file
        with open(weights_file, "rb") as f:
            header = np.fromfile(f, dtype=np.int32, count=4)  # First four are header values
            self.header_info = header  # Needed to write header when saving weights
            if seen is None:
                self.seen = header[3]  # number of images seen during training
            else:
                self.seen = seen
            weights = np.fromfile(f, dtype=np.float32)  # The rest are weights

        # Establish cutoff for loading backbone weights
        cutoff = None
        if "darknet.weights" in weights_file:
            cutoff = 13

        ptr = 0
        for i, (module_def, module) in enumerate(zip(modules_def, module_list)):
            if i == cutoff:
                break
            if module_def["type"] == "convolutional":
                conv_layer = module[0]
                if module_def["batch_normalize"]:
                    # Load BN bias, weights, running mean and running variance
                    bn_layer = module[1]
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
                else:
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
        for i, (module_def, module) in enumerate(zip(self.modules_def[:cutoff], self.module_list[:cutoff])):
            if module_def["type"] == "convolutional":
                conv_layer = module[0]
                # If batch norm, save bn first
                if module_def["batch_normalize"]:
                    bn_layer = module[1]
                    bn_layer.bias.data.cpu().numpy().tofile(fp)
                    bn_layer.weight.data.cpu().numpy().tofile(fp)
                    bn_layer.running_mean.data.cpu().numpy().tofile(fp)
                    bn_layer.running_var.data.cpu().numpy().tofile(fp)
                # save conv bias
                else:
                    conv_layer.bias.data.cpu().numpy().tofile(fp)
                # save conv weights
                conv_layer.weight.data.cpu().numpy().tofile(fp)

        fp.close()

        self.logger.print_log('Saved weights to '+fname)