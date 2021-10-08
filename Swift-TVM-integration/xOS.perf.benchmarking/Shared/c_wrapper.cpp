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

extern "C" void* createTVMInferWrapper()
{
    TVMInferWrapper* tvm_wrapper = new TVMInferWrapper();
    return (void*) tvm_wrapper;
}

extern "C" int doTVMinfer(void* handle, unsigned int* array, TVMClassificationResult* result)
{
    TVMInferWrapper* tvm_wrapper = (TVMInferWrapper*)handle;
    return tvm_wrapper->infer(array, result);
}

extern "C" int setTVMInputParams(void* handle, int batch, int channels, int height, int width)
{
    TVMInferWrapper* tvm_wrapper = (TVMInferWrapper*)handle;
    return tvm_wrapper->setParams(batch, channels, height, width);
}

extern "C" void removeTVMInferWrapper(void* handle)
{
    TVMInferWrapper* tvm_wrapper = (TVMInferWrapper*)handle;
    delete tvm_wrapper;
}
