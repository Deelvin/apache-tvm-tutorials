//
//

#ifndef ANDROIDDEMO_POSTPROCESSOR_H
#define ANDROIDDEMO_POSTPROCESSOR_H
#include "model.h"
// performs background removing step
template <typename T, typename std::enable_if<std::is_same_v<T,float> || std::is_same_v<T,float16_t>>::type* = nullptr>
class Postprocessor
{
public:
    Postprocessor()
    {};
    virtual ~Postprocessor() {};
    int convertToRGBA8(const tvm::runtime::NDArray& frame,
                       const tvm::runtime::NDArray& pha,
                       uint8_t* dataToSave,
                       uint8_t R = 120,
                       uint8_t G = 255,
                       uint8_t B = 155) {
        auto shape = frame.Shape();
        postprocessAndConvertToRGBA8((const T*)frame->data,
                                     (const T*)pha->data,
                                     dataToSave,
                                     shape[2],
                                     shape[3],
                                     R, G, B);
        return 0;
    };

    int trickyConvertToRGBA8(const tvm::runtime::NDArray& pha,
                       uint8_t* pData,
                       uint8_t R = 120,
                       uint8_t G = 255,
                       uint8_t B = 155) {
        auto shape = pha.Shape();
        trickyPostprocessAndConvertToRGBA8((const T*)pha->data,
                                     pData,
                                     shape[2],
                                     shape[3],
                                     R, G, B);
        return 0;
    };

protected:
};

#endif //ANDROIDDEMO_POSTPROCESSOR_H
