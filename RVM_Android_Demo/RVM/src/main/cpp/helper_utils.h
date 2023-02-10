//
//

#ifndef HELPER_UTILS_H
#define HELPER_UTILS_H

#include <arm_neon.h>

static inline void convertHalfChannel(uint16x8_t inpt, float32_t* pOutput, float32_t scale)
    {
    auto hu16 = vget_high_u16(inpt);
    auto lu16 = vget_low_u16(inpt);
    uint32x4_t u32x4srch = vmovl_u16(hu16);
    uint32x4_t u32x4srcl = vmovl_u16(lu16);
    vst1q_f32(pOutput + 0, vmulq_n_f32(vcvtq_f32_u32(u32x4srcl), scale));
    vst1q_f32(pOutput + 4, vmulq_n_f32(vcvtq_f32_u32(u32x4srch), scale));
}

static inline void convertChannel(uint8x16_t inpt, float32_t* pOutput, float32_t scale)
{
    uint16x8_t u16x8_l   = vmovl_u8(vget_low_u8(inpt));
    convertHalfChannel(u16x8_l, pOutput, scale);
    uint16x8_t u16x8_h   = vmovl_u8(vget_high_u8(inpt));
    convertHalfChannel(u16x8_h, pOutput + 8, scale);
}

static inline void convertRGBA8toCHW(uint8_t const*  pInput,
                                      uint32_t height,
                                      uint32_t width,
                                      float32_t* pOutput
                                      )
{
    if (pInput == nullptr ||
       pOutput == nullptr)
        return;
    float32_t scale = 1.0f/255.0f;
    uint32_t numPixels = height * width;
    uint32_t alignedPixels = (numPixels * 4) & 0xffffff40;
    uint8x16x4_t intlv_rgba;
    float32_t* R = &pOutput[0 * numPixels];
    float32_t* G = &pOutput[1 * numPixels];
    float32_t* B = &pOutput[2 * numPixels];
    int i = 0;
    for (; i < alignedPixels; i += 64) {
        intlv_rgba = vld4q_u8(pInput + i);
        convertChannel(intlv_rgba.val[0], R, scale);
        convertChannel(intlv_rgba.val[1], G, scale);
        convertChannel(intlv_rgba.val[2], B, scale);
        R += 16;
        G += 16;
        B += 16;
    }
    for (;i < numPixels; ++i) {
        *R = pInput[0] * scale;
        *G = pInput[1] * scale;
        *B = pInput[2] * scale;
        pInput += 4;
        R++;
        G++;
        B++;
    }
}

static inline void CONVERT_TO_CHANEL_FP16(uint8x16_t& inpt, float16_t* pOutput) {
    auto val_l = vmovl_u8(vget_low_u8(inpt)) << 2;
    auto val_h = vmovl_u8(vget_high_u8(inpt)) << 2;
    auto val_l_1 = val_l >> 8;
    val_l += val_l_1;
    auto val_h_1 = val_h >> 8;
    val_h += val_h_1;
    vst1q_f16(pOutput + 0, vcvtq_n_f16_u16(val_l, 10));
    vst1q_f16(pOutput + 8, vcvtq_n_f16_u16(val_h, 10));
}

static inline void convertRGBA8toCHW(uint8_t const*  pInput,
                                     uint32_t height,
                                     uint32_t width,
                                     float16_t* pOutput)
{
    if (pInput == nullptr ||
        pOutput == nullptr)
        return;
    uint32_t numPixels = height * width;
    uint32_t alignedPixels = (numPixels * 4) & 0xffffff40;
    uint8x16x4_t intlv_rgba;
    float16_t* R = &pOutput[0 * numPixels];
    float16_t* G = &pOutput[1 * numPixels];
    float16_t* B = &pOutput[2 * numPixels];
    int i = 0;
    for (; i < alignedPixels; i += 64) {
        intlv_rgba = vld4q_u8(pInput + i);
        CONVERT_TO_CHANEL_FP16(intlv_rgba.val[0], R);
        CONVERT_TO_CHANEL_FP16(intlv_rgba.val[1], G);
        CONVERT_TO_CHANEL_FP16(intlv_rgba.val[2], B);
        R += 16;
        G += 16;
        B += 16;
    }

    for (;i < numPixels; ++i) {
        float16_t scale = 1.f/255.f;
        *R = pInput[0] * scale;
        *G = pInput[1] * scale;
        *B = pInput[2] * scale;
        pInput += 4;
        R++;
        G++;
        B++;
    }
}

static inline uint8x16_t fillChannel(float32_t const*  chan_data,
                                     float32x4_t alpha1,
                                     float32x4_t alpha2,
                                     float32x4_t alpha3,
                                     float32x4_t alpha4,
                                     float32_t color)
{
    auto r1 = vld1q_f32(chan_data +  0);
    auto r2 = vld1q_f32(chan_data +  4);
    auto r3 = vld1q_f32(chan_data +  8);
    auto r4 = vld1q_f32(chan_data + 12);
    auto colorReg = vdupq_n_f32(color);
    auto res1 = vmlaq_f32(colorReg, r1, alpha1);

    auto res2 = vmlaq_f32(colorReg, r2, alpha2);
    auto res3 = vmlaq_f32(colorReg, r3, alpha3);
    auto res4 = vmlaq_f32(colorReg, r4, alpha4);

    res1 = vmlsq_f32(res1, alpha1, colorReg);
    res2 = vmlsq_f32(res2, alpha2, colorReg);
    res3 = vmlsq_f32(res3, alpha3, colorReg);
    res4 = vmlsq_f32(res4, alpha4, colorReg);

    res1 = vmulq_n_f32(res1, 255.f);
    res2 = vmulq_n_f32(res2, 255.f);
    res3 = vmulq_n_f32(res3, 255.f);
    res4 = vmulq_n_f32(res4, 255.f);

    // conversions fp32->u8
    uint32x4_t u32res1 = vcvtq_u32_f32(res1);
    uint32x4_t u32res2 = vcvtq_u32_f32(res2);

    uint32x4_t u32res3 = vcvtq_u32_f32(res3);
    uint32x4_t u32res4 = vcvtq_u32_f32(res4);

    uint16x8_t  u16L = vcombine_u16(vmovn_u32(u32res1), vmovn_u32(u32res2));
    uint16x8_t  u16H = vcombine_u16(vmovn_u32(u32res3), vmovn_u32(u32res4));

    return vcombine_u8(vqmovn_u16(u16L), vqmovn_u16(u16H));
}

static inline
uint8x16_t fillChannel(uint8x16_t chan_data,
                       float32x4_t alpha1,
                       float32x4_t alpha2,
                       float32x4_t alpha3,
                       float32x4_t alpha4,
                       uint8_t color)
{
    // conversions fp32->u8
    uint32x4_t u32res1 = vcvtq_u32_f32(alpha1 * 255);
    uint32x4_t u32res2 = vcvtq_u32_f32(alpha2 * 255);
    uint32x4_t u32res3 = vcvtq_u32_f32(alpha3 * 255);
    uint32x4_t u32res4 = vcvtq_u32_f32(alpha4 * 255);

    uint16x8_t alpha_u16L = vcombine_u16(vmovn_u32(u32res1), vmovn_u32(u32res2));
    uint16x8_t alpha_u16H = vcombine_u16(vmovn_u32(u32res3), vmovn_u32(u32res4));
    uint16x8_t sub_alpha_u16L = 255 - alpha_u16L;
    uint16x8_t sub_alpha_u16H = 255 - alpha_u16H;
    uint16x8_t val_l = vmovl_u8(vget_low_u8(chan_data)) * alpha_u16L + sub_alpha_u16L * color;
    uint16x8_t val_h = vmovl_u8(vget_high_u8(chan_data)) * alpha_u16H + sub_alpha_u16H * color;
    return vcombine_u8(vqmovn_u16((val_l >> 8)), vqmovn_u16((val_h >> 8)));
}

static const uint8_t ALPHA = 255;
static inline void postprocessAndConvertToRGBA8(float32_t const*  fgr_data,
                                                float32_t const*  pha_data,
                                                uint8_t*          pOutput,
                                                uint32_t height,
                                                uint32_t width,
                                                float R = 120.f,
                                                float G = 255.f,
                                                float B = 155.f) {
    if (fgr_data == nullptr ||
        pha_data == nullptr ||
        pOutput == nullptr) {
        return;
    }
    const float scale = 1.f/255.f;
    R *= scale;
    G *= scale;
    B *= scale;

    uint32_t numPixels = height * width;
    uint32_t alignedPixels = numPixels & 0xffffff40;
    float32_t const* pRdata = &fgr_data[0 * numPixels];
    float32_t const* pGdata = &fgr_data[1 * numPixels];
    float32_t const* pBdata = &fgr_data[2 * numPixels];
    int i = 0;
    uint8x16x4_t rgba;
    rgba.val[3] = vdupq_n_u8(ALPHA);
    for (; i < alignedPixels; i += 16) {
        auto alpha1 = vld1q_f32(pha_data +  0);
        auto alpha2 = vld1q_f32(pha_data +  4);
        auto alpha3 = vld1q_f32(pha_data +  8);
        auto alpha4 = vld1q_f32(pha_data + 12);
        rgba.val[0] = fillChannel(pRdata, alpha1, alpha2, alpha3, alpha4, R);
        rgba.val[1] = fillChannel(pGdata, alpha1, alpha2, alpha3, alpha4, G);
        rgba.val[2] = fillChannel(pBdata, alpha1, alpha2, alpha3, alpha4, B);
        vst4q_u8(pOutput, rgba);
        pRdata += 16;
        pGdata += 16;
        pBdata += 16;
        pha_data += 16;
        pOutput  += 64;
    }

    for (; i < numPixels; ++i) {
        auto fgr_r = *pRdata;
        auto fgr_g = *pGdata;
        auto fgr_b = *pBdata;
        auto alpha = *pha_data;
        auto out_r = 255 * (fgr_r * alpha + R * (1 - alpha));
        auto out_g = 255 * (fgr_g * alpha + G * (1 - alpha));
        auto out_b = 255 * (fgr_b * alpha + B * (1 - alpha));
        unsigned char r = (unsigned char)out_r;
        unsigned char g = (unsigned char)out_g;
        unsigned char b = (unsigned char)out_b;
        pOutput[0] = r;
        pOutput[1] = g;
        pOutput[2] = b;
        pOutput[3] = ALPHA;
        pOutput += 4;
        pRdata ++;
        pGdata ++;
        pBdata ++;
        pha_data ++;
    }
}

static inline
void trickyPostprocessAndConvertToRGBA8(float32_t const*  pha_data,
                                        uint8_t*          pOutput,
                                        uint32_t height,
                                        uint32_t width,
                                        uint8_t R = 120,
                                        uint8_t G = 255,
                                        uint8_t B = 155) {
    if (pha_data == nullptr ||
        pOutput == nullptr) {
        return;
    }

    uint32_t numPixels = height * width;
    uint32_t alignedPixels = numPixels & 0xffffff40;
    int i = 0;
    uint8x16x4_t rgba;
    for (; i < alignedPixels; i += 16) {
        rgba = vld4q_u8(pOutput);
        auto alpha1 = vld1q_f32(pha_data +  0);
        auto alpha2 = vld1q_f32(pha_data +  4);
        auto alpha3 = vld1q_f32(pha_data +  8);
        auto alpha4 = vld1q_f32(pha_data + 12);
        rgba.val[0] = fillChannel(rgba.val[0], alpha1, alpha2, alpha3, alpha4, R);
        rgba.val[1] = fillChannel(rgba.val[1], alpha1, alpha2, alpha3, alpha4, G);
        rgba.val[2] = fillChannel(rgba.val[2], alpha1, alpha2, alpha3, alpha4, B);
        vst4q_u8(pOutput, rgba);
        pha_data += 16;
        pOutput  += 64;
    }

    for (; i < numPixels; ++i) {
        uint8_t alpha = (*pha_data) * 255;
        uint32_t out_r = (pOutput[0] * alpha + R * (255 - alpha)) >> 8;
        uint32_t out_g = (pOutput[1] * alpha + G * (255 - alpha)) >> 8;
        uint32_t out_b = (pOutput[2] * alpha + B * (255 - alpha)) >> 8;
        unsigned char r = (unsigned char)out_r;
        unsigned char g = (unsigned char)out_g;
        unsigned char b = (unsigned char)out_b;
        pOutput[0] = r;
        pOutput[1] = g;
        pOutput[2] = b;
        pOutput += 4;
        pha_data ++;
    }
}

static inline uint8x16_t fillChannel(float16_t const*  chan_data,
                                     float16x8_t alpha1,
                                     float16x8_t alpha2,
                                     float16_t color)
{
    auto r1 = vld1q_f16(chan_data +  0);
    auto r2 = vld1q_f16(chan_data +  8);
    auto colorReg = vdupq_n_f16(color);
    auto res1 = vfmaq_f16(colorReg, r1, alpha1);
    auto res2 = vfmaq_f16(colorReg, r2, alpha2);

    res1 = vfmsq_f16(res1, alpha1, colorReg);
    res2 = vfmsq_f16(res2, alpha2, colorReg);

    res1 = vmulq_n_f16(res1, 255.f);
    res2 = vmulq_n_f16(res2, 255.f);

    // conversions fp16->u8
    uint16x8_t u16res1 = vcvtq_u16_f16(res1);
    uint16x8_t u16res2 = vcvtq_u16_f16(res2);

    return vcombine_u8(vqmovn_u16(u16res1), vqmovn_u16(u16res2));
}


static inline
void postprocessAndConvertToRGBA8(float16_t const*  fgr_data,
                                  float16_t const*  pha_data,
                                  uint8_t*          pOutput,
                                  uint32_t height,
                                  uint32_t width,
                                  float16_t R = 120.f,
                                  float16_t G = 255.f,
                                  float16_t B = 155.f) {
    if (fgr_data == nullptr ||
        pha_data == nullptr ||
        pOutput == nullptr) {
        return;
    }

    const float16_t scale = 1.f/255.f;
    R *= scale;
    G *= scale;
    B *= scale;

    uint32_t numPixels = height * width;
    uint32_t alignedPixels = numPixels & 0xffffff40;
    float16_t const* pRdata = &fgr_data[0 * numPixels];
    float16_t const* pGdata = &fgr_data[1 * numPixels];
    float16_t const* pBdata = &fgr_data[2 * numPixels];

    int i = 0;
    uint8x16x4_t rgba;
    rgba.val[3] = vdupq_n_u8(ALPHA);

    for (; i < alignedPixels; i += 16) {
        auto alpha1 = vld1q_f16(pha_data +  0);
        auto alpha2 = vld1q_f16(pha_data +  8);
        rgba.val[0] = fillChannel(pRdata, alpha1, alpha2, R);
        rgba.val[1] = fillChannel(pGdata, alpha1, alpha2, G);
        rgba.val[2] = fillChannel(pBdata, alpha1, alpha2, B);
        vst4q_u8(pOutput, rgba);
        pRdata += 16;
        pGdata += 16;
        pBdata += 16;
        pha_data += 16;
        pOutput  += 64;
    }

    for (; i < numPixels; ++i) {
        auto fgr_r = *pRdata;
        auto fgr_g = *pGdata;
        auto fgr_b = *pBdata;
        auto alpha = *pha_data;
        auto out_r = 255 * (fgr_r * alpha + R * (1 - alpha));
        auto out_g = 255 * (fgr_g * alpha + G * (1 - alpha));
        auto out_b = 255 * (fgr_b * alpha + B * (1 - alpha));
        unsigned char r = (unsigned char)out_r;
        unsigned char g = (unsigned char)out_g;
        unsigned char b = (unsigned char)out_b;
        pOutput[0] = r;
        pOutput[1] = g;
        pOutput[2] = b;
        pOutput[3] = ALPHA;
        pOutput += 4;
        pRdata ++;
        pGdata ++;
        pBdata ++;
        pha_data ++;
    }
}

static inline
void convertFP16toFP32(const float16_t* pFrom,
                       float32_t* pTo,
                       size_t count)
{
    size_t j = 0;
    uint32_t alignedPixels16 = count & 0xfffffff0;
    uint32_t alignedPixels8  = count & 0xfffffff8;

    for (;j < alignedPixels16; j += 16) {
        auto val1 = vld1q_f16(pFrom +  0);
        auto val2 = vld1q_f16(pFrom +  8);
        auto low_val1 = vget_low_f16(val1);
        auto high_val1 = vget_high_f16(val1);
        auto resFP32Low = vcvt_f32_f16(low_val1);
        auto resFP32High = vcvt_f32_f16(high_val1);
        vst1q_f32(pTo, resFP32Low);
        vst1q_f32(pTo + 4, resFP32High);

        auto low_val2 = vget_low_f16(val2);
        auto high_val2 = vget_high_f16(val2);
        resFP32Low = vcvt_f32_f16(low_val2);
        resFP32High = vcvt_f32_f16(high_val2);

        vst1q_f32(pTo + 8, resFP32Low);
        vst1q_f32(pTo + 12, resFP32High);

        pFrom += 16;
        pTo += 16;
    }
    for (;j < alignedPixels8; j += 8) {
        auto val1 = vld1q_f16(pFrom +  0);
        auto low_val1 = vget_low_f16(val1);
        auto high_val1 = vget_high_f16(val1);
        auto resFP32Low = vcvt_f32_f16(low_val1);
        auto resFP32High = vcvt_f32_f16(high_val1);
        vst1q_f32(pTo, resFP32Low);
        vst1q_f32(pTo + 4, resFP32High);

        pFrom += 8;
        pTo += 8;
    }
    for (;j < count; j++) {
        pTo[j] = float32_t(pFrom[j]);
    }
}

//static int xx = 0;
//static inline
//uint8x16_t fillChannel(uint8x16_t chan_data,
//                       float16x8_t alpha1,
//                       float16x8_t alpha2,
//                       uint8_t color)
//{
//    auto val_l = vmovl_u8(vget_low_u8(chan_data));
//    auto val_h = vmovl_u8(vget_high_u8(chan_data));
//    auto val_lF16 = vcvtq_n_f16_u16(val_l, 8);
//    auto val_hF16 = vcvtq_n_f16_u16(val_h, 8);
//
//    auto colorReg = vdupq_n_f16(color);
//    auto res1 = vfmaq_f16(colorReg, val_lF16, alpha1);
//    auto res2 = vfmaq_f16(colorReg, val_hF16, alpha2);
//
//    res1 = vfmsq_f16(res1, alpha1, colorReg);
//    res2 = vfmsq_f16(res2, alpha2, colorReg);
//
//    res1 = vmulq_n_f16(res1, 255.f);
//    res2 = vmulq_n_f16(res2, 255.f);
//
//    // conversions fp16->u8
//    uint16x8_t u16res1 = vcvtq_u16_f16(res1);
//    uint16x8_t u16res2 = vcvtq_u16_f16(res2);
//
//    return vcombine_u8(vqmovn_u16(u16res1), vqmovn_u16(u16res2));
//
//    // conversions fp16->u8
////    uint8_t alpha = (*pha_data) * 255;
////    uint32_t out_r = (pOutput[0] * alpha + R * (255 - alpha)) >> 8;
////    uint32_t out_g = (pOutput[1] * alpha + G * (255 - alpha)) >> 8;
////    uint32_t out_b = (pOutput[2] * alpha + B * (255 - alpha)) >> 8;
////    if (xx == 0) {
////        alpha1[0] = 1.0;
////    }
////    uint16x8_t u16res1 = vcvtq_n_u16_f16(alpha1, 8);
////    uint16x8_t u16res2 = vcvtq_n_u16_f16(alpha2, 8);
////    if (xx == 0) {
////        std::cout << alpha1[0] << ", ";
////        std::cout << alpha1[1] << ", ";
////        std::cout << alpha1[2] << ", ";
////        std::cout << alpha1[3] << ", ";
////        std::cout << alpha1[4] << ", ";
////        std::cout << alpha1[5] << ", ";
////        std::cout << alpha1[6] << ", ";
////        std::cout << alpha1[7] << "\n";
////
////        std::cout << u16res1[0] << ", ";
////        std::cout << u16res1[1] << ", ";
////        std::cout << u16res1[2] << ", ";
////        std::cout << u16res1[3] << ", ";
////        std::cout << u16res1[4] << ", ";
////        std::cout << u16res1[5] << ", ";
////        std::cout << u16res1[6] << ", ";
////        std::cout << u16res1[7] << "\n";
////
////        xx = 1;
////    }
////    uint16x8_t  sub_alpha_u16L = 255 - u16res1;
////    uint16x8_t  sub_alpha_u16H = 255 - u16res2;
////    uint16x8_t val_l = ((vmovl_u8(vget_low_u8(chan_data)) * u16res1) >> 8) + ((sub_alpha_u16L * color) >> 8);
////    uint16x8_t val_h = ((vmovl_u8(vget_high_u8(chan_data)) * u16res2) >>8) + ((sub_alpha_u16H * color) >> 8);
////    return vcombine_u8(vqmovn_u16(val_l), vqmovn_u16(val_h));
//}
//
//static inline
//void trickyPostprocessAndConvertToRGBA8(float16_t const*  pha_data,
//                                        uint8_t*          pOutput,
//                                        uint32_t height,
//                                        uint32_t width,
//                                        uint8_t R = 120,
//                                        uint8_t G = 255,
//                                        uint8_t B = 155) {
//    if (pha_data == nullptr ||
//        pOutput == nullptr) {
//        return;
//    }
//
//    uint32_t numPixels = height * width;
//    uint32_t alignedPixels = numPixels & 0xffffff40;
//
//    int i = 0;
//    uint8x16x4_t rgba;
//
//    for (; i < alignedPixels; i += 16) {
//        rgba = vld4q_u8(pOutput);
//        auto alpha1 = vld1q_f16(pha_data +  0);
//        auto alpha2 = vld1q_f16(pha_data +  8);
//        rgba.val[0] = fillChannel(rgba.val[0], alpha1, alpha2, R);
//        rgba.val[1] = fillChannel(rgba.val[1], alpha1, alpha2, G);
//        rgba.val[2] = fillChannel(rgba.val[2], alpha1, alpha2, B);
//        vst4q_u8(pOutput, rgba);
//        pha_data += 16;
//        pOutput  += 64;
//    }
//
//    for (; i < numPixels; ++i) {
//        uint8_t alpha = (*pha_data) * 255;
//        uint32_t out_r = (pOutput[0] * alpha + R * (255 - alpha)) >> 8;
//        uint32_t out_g = (pOutput[1] * alpha + G * (255 - alpha)) >> 8;
//        uint32_t out_b = (pOutput[2] * alpha + B * (255 - alpha)) >> 8;
//        unsigned char r = (unsigned char)out_r;
//        unsigned char g = (unsigned char)out_g;
//        unsigned char b = (unsigned char)out_b;
//        pOutput[0] = r;
//        pOutput[1] = g;
//        pOutput[2] = b;
//        pOutput += 4;
//        pha_data ++;
//    }
//}

#endif //HELPER_UTILS_H
