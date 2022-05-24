import json
import os.path as osp

def rd_JSONInfo(path, input_shape, nclasses, number_convolutional_layers, casenet, weight_decay, momentum, optimizer,
                use3D, datatype, softlabel, dropout, nofreeze, loss_weights, ndense_units,
                margin, dynmargin, hardnegs, mergefun, batchsize, lr, filters_size, filters_numbers):
    data = {}

    data["model"] = {}
    data["model"]["inputShape"] = input_shape
    data["model"]["nClasses"] = nclasses
    data["model"]["numberConvolutionLayers"] = number_convolutional_layers
    data["model"]["casenet"] = casenet
    data["model"]["3D"] = use3D
    data["model"]["ndenseUnits"] = ndense_units
    data["model"]["margin"] = margin
    data["model"]["filtersSize"] = filters_size
    data["model"]["filtersNumbers"] = filters_numbers


    data["train"] = {}
    data["train"]["weightDecay"] = weight_decay
    data["train"]["momentum"] = momentum
    data["train"]["optimizer"] = optimizer
    data["train"]["dropout"] = dropout
    data["train"]["lossWeights"] = loss_weights
    data["train"]["dynMargin"] = dynmargin
    data["train"]["hardNegs"] = hardnegs
    data["train"]["mergeFun"] = mergefun
    data["train"]["batchSize"] = batchsize
    data["train"]["lr"] = lr


    data["data"] = {}
    data["data"]["dataType"] = datatype
    data["data"]["softLabel"] = softlabel
    data["data"]["noFreeze"] = nofreeze



    with open(osp.join(path, 'info.json'), 'w') as outfile:
        json.dump(data, outfile)
