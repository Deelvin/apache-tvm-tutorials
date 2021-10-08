/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

#include "tvm_bridge.h"

#include "tvm/runtime/module.h"
#include "tvm/runtime/packed_func.h"
#include "tvm/runtime/registry.h"

#include <stdio.h>

typedef std::chrono::high_resolution_clock Time;
typedef std::chrono::nanoseconds ns;

TVMInferWrapper::TVMInferWrapper() {
    m_height  = 224;
    m_width   = 224;
    m_batch   = 1;
    m_channels = 3;
}

int TVMInferWrapper::setParams(int batch, int channels, int height, int width) {
    m_height  = height;
    m_width   = width;
    m_batch   = batch;
    m_channels = channels;

    return 0;
}


int TVMInferWrapper::infer(unsigned int* data, TVMClassificationResult* result) {

    // Use the C++ API
    DLDevice ctx = {kDLCPU, 0};

    const std::string input_model{"compiled_model.dylib"};

    tvm::runtime::Module mod_factory = tvm::runtime::Module::LoadFromFile(input_model);

    // create the graph runtime module
    tvm::runtime::Module gmod = mod_factory.GetFunction("default")(ctx);
    tvm::runtime::PackedFunc set_input = gmod.GetFunction("set_input");
    tvm::runtime::PackedFunc get_input = gmod.GetFunction("get_input");
    tvm::runtime::PackedFunc get_output = gmod.GetFunction("get_output");
    tvm::runtime::PackedFunc run = gmod.GetFunction("run");
    
    uint8_t* input_buffer = (uint8_t*)data;

    tvm::runtime::NDArray x = tvm::runtime::NDArray::Empty({1, m_channels, m_width, m_height}, DLDataType{kDLFloat, 32, 1}, ctx);
    
    // Prepare input data by reordering from ARGB NHWC to RGB NCHW and normilization
    float std[] = {0.229f, 0.224f, 0.225f};
    float mean[]  = {0.485f, 0.456f, 0.406f};
    
    int nielements  = m_channels * m_width * m_height;
    
    float *tmpBuffer = (float *)malloc(nielements * sizeof(size_t));
    for (size_t sd = 0; sd < m_width * m_height; sd++) {
        for (size_t channel = 0; channel < m_channels; channel++) {
            size_t pidx = channel * m_width * m_height + sd;
            size_t idx = sd * 4 + 1; // +1 is needed to handle ARGB case and skip A
            tmpBuffer[pidx] = (input_buffer[idx + channel] / 255.0f - mean[channel]) / std[channel];
        }
    }
    
    for (size_t j = 0; j < nielements; j++) {
        static_cast<float*>(x->data)[j] = tmpBuffer[j];
    }
    free(tmpBuffer);
    
    // set the right input
    set_input(0, x);
    TVMSynchronize(ctx.device_type, ctx.device_id, nullptr);

    int niterations = 100;

    Time::time_point time0 = Time::now();
    
    for (size_t i = 0; i < niterations; i++) {
        run();
        TVMSynchronize(ctx.device_type, ctx.device_id, nullptr);
    }

    Time::time_point time1 = Time::now();

    tvm::runtime::NDArray y = tvm::runtime::NDArray::Empty({this->m_batch, 1000}, DLDataType{kDLFloat, 32, 1}, ctx);
    get_output(0, y);

    // Looking for the Top5 values and print them in format
    // class_id     probability/value of the latest tensor in case of softmax absence
    const float* oData = static_cast<float*>(y->data);
    std::map<float, size_t> ordered;
    for (size_t j = 0; j < 1000; j++) {
        ordered[oData[j]] = j;
    }

    int s = 0, topK = 5;
    std::cout << std::fixed << std::setprecision(2);
    for (auto it = ordered.crbegin(); it != ordered.crend() && s < topK; it++, s++) {
        std::cout << "Top"<< s+1 << " class#" << it->second << ": " << it->first*100 << std::endl;
    }

    auto it = ordered.crbegin();
    
    float ms = std::chrono::duration_cast<ns>(time1 - time0).count() * 0.000001;
    
    result->class_id = it->second;
    result->probability = it->first;
    result->performance = niterations * 1000./ms;
    
    return 0;
}
