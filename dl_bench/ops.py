from dl_bench.utils import Benchmark, RandomInfDataset

import torch

from collections import OrderedDict
import numpy as np

import torch.nn as nn



def summary(model, input_size, batch_size=-1, device=torch.device('cuda:0'), dtypes=None):
    result, params_info, summary = summary_string(
        model, input_size, batch_size, device, dtypes)
    print(result)

    return params_info, summary


def summary_string(model, input_size, batch_size=-1, device=torch.device('cuda:0'), dtypes=None):
    if dtypes == None:
        dtypes = [torch.FloatTensor]*len(input_size)

    summary_str = ''

    def register_hook(module):
        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)

            m_key = "%s-%i" % (class_name, module_idx + 1)
            summary[m_key] = OrderedDict()
            summary[m_key]["module"] = module
            summary[m_key]["input_shape"] = list(input[0].size())
            summary[m_key]["input_shape"][0] = batch_size
            if isinstance(output, (list, tuple)):
                summary[m_key]["output_shape"] = [
                    [-1] + list(o.size())[1:] for o in output
                ]
            else:
                summary[m_key]["output_shape"] = list(output.size())
                summary[m_key]["output_shape"][0] = batch_size

            params = 0
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                summary[m_key]["trainable"] = module.weight.requires_grad
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                params += torch.prod(torch.LongTensor(list(module.bias.size())))
            summary[m_key]["nb_params"] = params

        if (
            not isinstance(module, nn.Sequential)
            and not isinstance(module, nn.ModuleList)
        ):
            hooks.append(module.register_forward_hook(hook))

    # multiple inputs to the network
    if isinstance(input_size, tuple):
        input_size = [input_size]

    # batch_size of 2 for batchnorm
    x = [torch.rand(2, *in_size).type(dtype).to(device=device)
         for in_size, dtype in zip(input_size, dtypes)]

    # create properties
    summary = OrderedDict()
    hooks = []

    # register hook
    model.apply(register_hook)

    # make a forward pass
    # print(x.shape)
    model(*x)

    # remove these hooks
    for h in hooks:
        h.remove()

    summary_str += "----------------------------------------------------------------" + "\n"
    line_new = "{:>20}  {:>25}  {:>25} {:>15}".format(
        "Layer (type)", "Input Shape", "Output Shape", "Param #")
    summary_str += line_new + "\n"
    summary_str += "================================================================" + "\n"
    total_params = 0
    total_output = 0
    trainable_params = 0
    for layer in summary:
        # input_shape, output_shape, trainable, nb_params
        line_new = "{:>20}  {:>25}  {:>25} {:>15}".format(
            layer,
            str(summary[layer]["input_shape"]),
            str(summary[layer]["output_shape"]),
            "{0:,}".format(summary[layer]["nb_params"]),
        )
        total_params += summary[layer]["nb_params"]

        total_output += np.prod(summary[layer]["output_shape"])
        if "trainable" in summary[layer]:
            if summary[layer]["trainable"] == True:
                trainable_params += summary[layer]["nb_params"]
        summary_str += line_new + "\n"

    # assume 4 bytes/number (float on cuda).
    total_input_size = abs(np.prod(sum(input_size, ()))
                           * batch_size * 4. / (1024 ** 2.))
    total_output_size = abs(2. * total_output * 4. /
                            (1024 ** 2.))  # x2 for gradients
    total_params_size = abs(total_params * 4. / (1024 ** 2.))
    total_size = total_params_size + total_output_size + total_input_size

    summary_str += "================================================================" + "\n"
    summary_str += "Total params: {0:,}".format(total_params) + "\n"
    summary_str += "Trainable params: {0:,}".format(trainable_params) + "\n"
    summary_str += "Non-trainable params: {0:,}".format(total_params -
                                                        trainable_params) + "\n"
    summary_str += "----------------------------------------------------------------" + "\n"
    summary_str += "Input size (MB): %0.2f" % total_input_size + "\n"
    summary_str += "Forward/backward pass size (MB): %0.2f" % total_output_size + "\n"
    summary_str += "Params size (MB): %0.2f" % total_params_size + "\n"
    summary_str += "Estimated Total Size (MB): %0.2f" % total_size + "\n"
    summary_str += "----------------------------------------------------------------" + "\n"
    # return summary
    return summary_str, (total_params, trainable_params), summary

class Layers:
    def __init__(self) -> None:
        pass

    def get_convs_from_resnet(self):
        from torchvision.models import resnet50, resnet18, ResNet
        from torchvision.models.resnet import Bottleneck, BasicBlock

        resnet = resnet50()
        resnet.eval()
        _, summ = summary(resnet, ( 3, 224, 224), batch_size=-1, device=torch.device('cpu'))

        convs = []
        in_shs =[]
        import copy
        for layer in summ:
            module = summ[layer]["module"]
            if isinstance(module, nn.Conv2d):
                in_sh = summ[layer]["input_shape"]
                in_sh = in_sh[1:4]
                convs.append(module)
                in_shs.append(in_sh)
        self.convs = convs
        self.in_shs = in_shs



class Conv2dNoPaddingModule(torch.nn.Module):

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.conv = torch.nn.Conv2d(2, 10, 3, bias=False)
        self.train(False)

    def forward(self, x):
        return self.conv(x)

layers = Layers()
layers.get_convs_from_resnet()

def get_op(name):
    name2model = {
        "conv210": Conv2dNoPaddingModule,
    }

    for i in range(len(layers.in_shs)):
        def factory(ind):
            return lambda: layers.convs[ind]
        name2model["conv_" + str(i)] = factory(i)

    if name in name2model:
        return name2model[name]()
    else:
        raise ValueError(f"Unknown name {name}")


class OpsBenchmark(Benchmark):
    def __init__(self, params) -> None:
        batch_size = int(params.get("batch_size", 1024))

        name = params.get("name", "conv210")
        if name.split("_")[0] == "conv":
            in_shs = layers.in_shs
            in_shape = tuple(in_shs[int(name.split("_")[1])])
            # print(tuple(in_shape), " type: ", type(in_shape))
        else:
            in_shape = (2, 10, 20)
        min_batches = 10
        DATASET_SIZE = max(10_240, batch_size * min_batches)
        dataset = RandomInfDataset(DATASET_SIZE, in_shape)
        # import sys
        # sys.exit(0)
        net = get_op(name=name)
        
        super().__init__(
            net=net, in_shape=in_shape, dataset=dataset, batch_size=batch_size
        )