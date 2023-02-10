
// This is automatically generated header (see prepare_model.py for details)

#pragma once

#include <vector>
#include <string>
#include <dlpack/dlpack.h>

struct item {
  std::string name;
  const std::vector<int64_t> shape;
  std::string dtype;
};

static const std::vector<item> s_model_inputs = {
    {"inp0", {1,3,720,1280,}, "float32" },
    {"rec0", {1,16,144,256,}, "float32" },
    {"rec1", {1,20,72,128,}, "float32" },
    {"rec2", {1,40,36,64,}, "float32" },
    {"rec3", {1,64,18,32,}, "float32" },

};

static const std::string s_lib_name = "android.hg_demo_mobilenetv3.float16.atvm.720.so";
constexpr bool USE_GE = true;
constexpr bool USE_FP16 = true;
constexpr DLDeviceType DL_DEVICE_TYPE = kDLOpenCL;

using RVMInputType = float;

#if defined __x86_64__
template <class T>
struct check_fp16
{
    static_assert(USE_FP16 == false, "There is no device to test FP16 model on x86_64 arch for now!");
};
#endif


