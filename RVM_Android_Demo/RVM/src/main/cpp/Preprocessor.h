//
//

#ifndef ANDROIDDEMO_PREPROCESSOR_H
#define ANDROIDDEMO_PREPROCESSOR_H

#include <vector>
#include <type_traits>
#include "helper_utils.h"
#include "model.h"

//conversion RGBA to [1, 3, H, W] tensor

template<typename T, typename std::enable_if<std::is_same_v<T,float> || std::is_same_v<T,float16_t>>::type* = nullptr>
class Preprocessor
{
public:
    Preprocessor()
    {
        tvm::Device cpu_ctx{kDLCPU, 0};
        out_tensor_ = gen_zero_data<T>(s_model_inputs[0].shape, cpu_ctx, s_model_inputs[0].dtype);
    }

    Preprocessor(int64_t height, int64_t width)
    {
        tvm::Device cpu_ctx{kDLCPU, 0};
        std::string dtype;
        if constexpr (std::is_same_v<T,float16_t>) {
            dtype = "float16";
        } else{
            dtype = "float32";
        }
        out_tensor_ = gen_zero_data<T>({1, 3, height, width}, cpu_ctx, dtype);
    }

    virtual ~Preprocessor() {};

    int evaluateRGBA8(const void* pData){
        auto shape = out_tensor_.Shape();
        convertRGBA8toCHW((uint8_t const*)pData,
                          shape[2],
                          shape[3],
                          (T* )out_tensor_->data);
        return 0;
    };
    const tvm::runtime::NDArray& getResult() {
        return out_tensor_;
    }
protected:
    tvm::runtime::NDArray out_tensor_;
};

#endif //ANDROIDDEMO_PREPROCESSOR_H
