import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import tqdm
import time

from modules import *
from utils.computation import *
from utils.utils import log_train_progress, show_eval_result

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
    def __init__(self, options, logger):
        super(BaseModel, self).__init__()
        self.options = options
        self.logger = logger
        self.modules_def = self.parse_model_cfg(options.model_cfg)
        self.hyper_parameters, self.module_list = self.get_module_list()

        self.optimizer = optim.SGD(self.module_list.parameters(),
                                   lr=float(self.hyper_parameters['learning_rate']),
                                   momentum=float(self.hyper_parameters['momentum']),
                                   weight_decay=float(self.hyper_parameters['decay']))

    def forward(self, inputs, targets=None):
        outputs = None
        if targets is not None:
            loss = 0
            for i, (module_def, module) in enumerate(zip(self.modules_def, self.module_list)):
                if module_def["type"] in ["convolutional", "upsample", "maxpool"]:
                    inputs = module(inputs)
                elif module_def["type"] == "region":
                    outputs, region_loss = module[0](inputs, targets)
                    loss += region_loss
                else:
                    print('unknown type %s' % (module_def['type']))
            outputs = outputs.detach().cpu()
            return outputs, loss
        else:
            with torch.no_grad():
                for i, (module_def, module) in enumerate(zip(self.modules_def, self.module_list)):
                    if module_def["type"] in ["convolutional", "upsample", "maxpool"]:
                        inputs = module(inputs)
                    elif module_def["type"] == "region":
                        outputs, _ = module[0](inputs)
                    else:
                        print('unknown type %s' % (module_def['type']))
            outputs = outputs.detach().cpu()
            return outputs

    def train(self, options, train_dataloader, eval_dataloader):
        for epoch in range(options.epochs):
            start_time = time.time()
            self.set_train_state()
            for batch_i, (imgs, targets, img_path) in enumerate(train_dataloader):
                imgs = Variable(imgs.type(torch.cuda.FloatTensor))
                targets = Variable(targets.type(torch.cuda.FloatTensor), requires_grad=False)
                outputs, loss = self.forward(imgs, targets)
                loss.backward()
                self.optimizer.zero_grad()
                self.optimizer.step()
                log_train_progress(epoch, options.epochs, batch_i, len(train_dataloader), start_time,
                                   self.module_list[-1][0].metrics, self.logger)

            if epoch % options.eval_interval == options.eval_interval - 1:
                self.logger.print_log("\n---- Evaluating Model ----")
                self.eval(eval_dataloader)

    def eval(self, dataloader):
        self.set_eval_state()
        metrics = []
        labels = []
        for batch_i, (imgs, targets, img_path) in enumerate(tqdm.tqdm(dataloader, desc="Detecting objects")):
            labels += targets[:, 1].tolist()
            imgs = Variable(imgs.type(torch.cuda.FloatTensor), requires_grad=False)
            # Rescale target
            targets[:, 2:] = xywh2xyxy(targets[:, 2:])
            targets[:, 2:] *= int(self.hyper_parameters['width'])

            outputs = self.forward(imgs)  # B,845,25
            predictions = non_max_suppression(outputs, self.options.conf_thresh, self.options.nms_thresh)
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

    def get_module_list(self):
        """
            Constructs module list of layer blocks from module configuration in modules_def
        """
        hyper_parameters = self.modules_def.pop(0)
        output_filters = [int(hyper_parameters["channels"])]
        module_list = nn.ModuleList()
        for module_i, module_def in enumerate(self.modules_def):
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

            elif module_def["type"] == "region":
                region = RegionLoss(module_def, hyper_parameters)
                modules.add_module(f"region_{module_i}", region)

            else:
                print('unknown type %s' % (module_def['type']))

            modules = modules.cuda()
            module_list.append(modules)

        return hyper_parameters, module_list

    def load_weights(self, weights_path):
        """Parses and loads the weights stored in 'weights_path'"""

        # Open the weights file
        with open(weights_path, "rb") as f:
            header = np.fromfile(f, dtype=np.int32, count=4)  # First four are header values
            weights = np.fromfile(f, dtype=np.float32)  # The rest are weights

        # Establish cutoff for loading backbone weights
        cutoff = None
        if "darknet53.conv.74" in weights_path:
            cutoff = 75

        ptr = 0
        for i, (module_def, module) in enumerate(zip(self.modules_def, self.module_list)):
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