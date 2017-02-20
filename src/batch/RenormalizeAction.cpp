// Copyright kikaxa 2017
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#include "RenormalizeAction.hpp"
#include "normalize/NormalizationHelper.h"
#include "normalize/NormalizationLayer.h"
#include "net/NeuralNet.h"
#include <stdio.h>


RenormalizeAction::RenormalizeAction(NetAction *_wrap, RenormalizeActionConfig _config) :
    wrap(_wrap), config(_config)
{

}
RenormalizeAction::~RenormalizeAction()
{
    delete wrap;
}

void RenormalizeAction::run(Trainable *_net, int epoch, int batch, float const*const batchData, int const*const batchLabels)
{
    float translate;
    float scale;
    if (config.normalization == "stddev") {
        float mean, stdDev;
        NormalizationHelper::getMeanAndStdDev(batchData, config.size * _net->getInputCubeSize(), &mean, &stdDev);
        //std::cerr << " image stats mean " << mean << " stdDev " << stdDev << std::endl;
        translate = - mean;
        scale = 1.0f / stdDev / config.normalizationNumStds;
    } else if (config.normalization == "maxmin") {
        float mean, stdDev;
        NormalizationHelper::getMinMax(batchData, config.size * _net->getInputCubeSize(), &mean, &stdDev);
        translate = - mean;
        scale = 1.0f / stdDev;
    } else {
        std::cerr << "Error: Unknown normalization: " << config.normalization << std::endl;
        return;
    }

    NeuralNet *net = static_cast<NeuralNet *>(_net);
    for (int layerId = 0; layerId < net->getNumLayers(); layerId++) {
        Layer *layer = net->getLayer(layerId);
        if (layer->getClassName() == "NormalizationLayer")
        {
            NormalizationLayer *norm = static_cast<NormalizationLayer *>(layer);
            norm->scale = scale;
            norm->translate = translate;
            break;
        }
    }

    wrap->run(_net, epoch, batch, batchData, batchLabels);
}


#if 0 //TODO is Action2 used anywhere?
RenormalizeAction2::RenormalizeAction2(NetAction2 *_wrap, RenormalizeActionConfig _config) :
    wrap(_wrap), config(_config)
{

}
RenormalizeAction2::~RenormalizeAction2()
{
    delete wrap;
}

void RenormalizeAction2::run(Trainable *_net, int epoch, int batch, InputData *inputData, OutputData *outputData)
{
    float translate;
    float scale;
    if (config.normalization == "stddev") {
        float mean, stdDev;
        NormalizationHelper::getMeanAndStdDev(inputData->inputs, config.size * inputData->inputCubeSize, &mean, &stdDev);
        //std::cerr << " image stats mean " << mean << " stdDev " << stdDev << std::endl;
        translate = - mean;
        scale = 1.0f / stdDev / config.normalizationNumStds;
    } else if (config.normalization == "maxmin") {
        float mean, stdDev;
        NormalizationHelper::getMinMax(inputData->inputs, config.size * inputData->inputCubeSize, &mean, &stdDev);
        translate = - mean;
        scale = 1.0f / stdDev;
    } else {
        std::cerr << "Error: Unknown normalization: " << config.normalization << std::endl;
        return;
    }

    NeuralNet *net = static_cast<NeuralNet *>(_net);
    for (int layerId = 0; layerId < net->getNumLayers(); layerId++) {
        Layer *layer = net->getLayer(layerId);
        if (layer->getClassName() == "NormalizationLayer")
        {
            NormalizationLayer *norm = static_cast<NormalizationLayer *>(layer);
            norm->scale = scale;
            norm->translate = translate;
            break;
        }
    }

    wrap->run(_net, epoch, batch, inputData, outputData);
}
#endif
