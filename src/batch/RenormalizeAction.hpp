// Copyright kikaxa 2017
//
// This Source Code Form is subject to the terms of the Mozilla Public License, 
// v. 2.0. If a copy of the MPL was not distributed with this file, You can 
// obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "NetAction.h"
#include "NetAction2.h"

struct DeepCL_EXPORT RenormalizeActionConfig {
    int size;
    float normalizationNumStds;
    std::string normalization;
};

class DeepCL_EXPORT RenormalizeAction : public NetAction {
public:
    RenormalizeActionConfig config;
    NetAction *wrap;
    RenormalizeAction(NetAction *wrap, RenormalizeActionConfig config); //takes owhership
    virtual ~RenormalizeAction();
    //trainable must be NeuralNet
    virtual void run(Trainable *net, int epoch, int batch, float const*const batchData, int const*const batchLabels);
};

#if 0 //TODO is Action2 used anywhere?
class DeepCL_EXPORT RenormalizeAction2 : public NetAction2 { //TODO support for replacement of NetLearnAction2(extensions, use by ptr)
public:
    RenormalizeActionConfig config;
    NetAction2 *wrap;
    RenormalizeAction2(NetAction2 *wrap, RenormalizeActionConfig config); //takes owhership
    virtual ~RenormalizeAction2();
    //trainable must be NeuralNet
    virtual void run(Trainable *net, int epoch, int batch, InputData *inputData, OutputData *outputData);
};
#endif
