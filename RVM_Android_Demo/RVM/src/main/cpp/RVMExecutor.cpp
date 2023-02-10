//
//
#include "RVMExecutor.h"
#include "helper_utils.h"
#include <android/log.h>
#define APPNAME "RVM.Executor"
#define LOGGER(...) { \
    __android_log_print(ANDROID_LOG_VERBOSE, APPNAME, __VA_ARGS__); \
}

RVMExecutor::RVMExecutor(const std::string& libPath)
{
    tvm::Device ctx{DL_DEVICE_TYPE, 0};
    const std::string lib_name = s_lib_name;
    std::string exec_name;
    std::string consts_name;
    if constexpr (USE_GE == false) {
        consts_name = "consts_720_1280";
        exec_name = "vm_exec_code_720_1280.ro";
    }
    LOGGER("model name is %s", lib_name.c_str())
    LOGGER("model name is %s", libPath.c_str())
    executable_.init(libPath + lib_name, libPath + consts_name,
                     libPath + exec_name, ctx);
    model_ = std::make_unique<Model<USE_GE>>(executable_);
    if (model_ == nullptr) {
        LOGGER("MODEL failed");
        return;
    }
//        LOGGER("MODEL created");
//    } else {
//        LOGGER("MODEL failed");
//    }
    init_output_objects(outputs_);
    completeInitialization_ = true;
}

RVMExecutor::~RVMExecutor()
{

}
