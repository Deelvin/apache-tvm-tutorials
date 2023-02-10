#include <gtest/gtest.h>
#include <chrono>
#include "Preprocessor.h"
#include "Postprocessor.h"
//#include <opencv2/dnn/dnn.hpp>

// use large value for more accurate test
static const uint32_t COUNT = 300;

TEST(Preprocessor, InitFP32) {
    auto preprocessor = Preprocessor<float>(128, 128);
    auto pData = preprocessor.getResult();
    ASSERT_NE(pData->data, nullptr);
}

TEST(Preprocessor, InitFP16) {
    auto preprocessor = Preprocessor<float16_t>(128, 128);
    auto pData = preprocessor.getResult();
    ASSERT_NE(pData->data, nullptr);
}

TEST(Postrocessor, InitFP32) {
    auto postrocessor = Postprocessor<float>();
}

TEST(Postrocessor, InitFP16) {
    auto postrocessor = Postprocessor<float16_t>();
}

static void fill_rgba(std::vector<uint8_t>& rgba)
{
    for (uint32_t i = 0; i < rgba.size(); i += 4) {
        rgba[i + 0] = (rand() % 256);
        rgba[i + 1] = (rand() % 256);
        rgba[i + 2] = (rand() % 256);
        rgba[i + 3] = (rand() % 256);
    }
}

template<typename T>
static void fill_float(tvm::runtime::NDArray& data)
{
    auto shape = data.Shape();
    int64_t sz = 1;
    for (auto i : shape) {
        sz *= i;
    }
    T* pData = static_cast<T*>(data->data);
    for (uint32_t i = 0; i < sz; i++) {
        float val = rand() / float(RAND_MAX);
        pData[i] = T(val);
    }
}

TEST(Preprocessor, Run) {
    uint32_t height = 64;
    uint32_t width  = 128;
    srand(377);
    auto sz = height * width;
    std::vector<uint8_t> rgba(sz * 4);

    for (uint32_t i = 0; i < rgba.size(); i += 4) {
        rgba[i + 0] = 1;
        rgba[i + 1] = 33;
        rgba[i + 2] = 76;
        rgba[i + 3] = 129;
    }

    auto preprocessor = Preprocessor<float>(height, width);
    preprocessor.evaluateRGBA8(rgba.data());
    float* pData = static_cast<float*>(preprocessor.getResult()->data);
    auto pR = pData;
    auto pG = pR + sz;
    auto pB = pG + sz;

    auto ref_R = pR[0];
    auto ref_G = pG[0];
    auto ref_B = pB[0];
    ASSERT_NE(ref_R, ref_G);
    ASSERT_NE(ref_R, ref_B);

    for (auto i = 0; i < sz; ++i) {
        ASSERT_FLOAT_EQ(ref_R, pR[i]);
        ASSERT_FLOAT_EQ(ref_G, pG[i]);
        ASSERT_FLOAT_EQ(ref_B, pB[i]);
    }
}

using namespace std::chrono;

// height, width
using DimsDef = std::tuple<uint32_t, uint32_t>;

class PerfCustom :public ::testing::TestWithParam<DimsDef> {
};

class PerfCustomFP16 :public ::testing::TestWithParam<DimsDef> {
};

class PerfOpenCV :public ::testing::TestWithParam<DimsDef> {
};

class PerfPostprocessorFP32 :public ::testing::TestWithParam<DimsDef> {
};

class PerfTrickyPostprocessorFP32 :public ::testing::TestWithParam<DimsDef> {
};

class PerfTrickyPostprocessorFP16 :public ::testing::TestWithParam<DimsDef> {
};

class PerfPostprocessorFP16 :public ::testing::TestWithParam<DimsDef> {
};

TEST_P(PerfCustom, PerfCustomTest) {
    srand(124);
    auto dims = GetParam();
    uint32_t height = std::get<0>(dims);
    uint32_t width  = std::get<1>(dims);
    auto sz = height * width;
    std::vector<uint8_t> rgba(sz * 4);

    fill_rgba(rgba);

    auto preprocessor = Preprocessor<float>(height, width);
    high_resolution_clock::time_point start = high_resolution_clock::now();
    for (auto i = 0; i < COUNT; ++i) {
        preprocessor.evaluateRGBA8(rgba.data());
    }
    high_resolution_clock::time_point end = high_resolution_clock::now();
    auto elapsed = duration_cast<microseconds>(end - start).count();
    std::cout << height << "x" << width << " "<< elapsed/1000./COUNT << " ms.\n";
}

TEST_P(PerfCustomFP16, PerfCustomTest) {
    srand(9);
    auto dims = GetParam();
    uint32_t height = std::get<0>(dims);
    uint32_t width  = std::get<1>(dims);
    auto sz = height * width;
    std::vector<uint8_t> rgba(sz * 4);

    fill_rgba(rgba);

    auto preprocessor = Preprocessor<float16_t>(height, width);
    high_resolution_clock::time_point start = high_resolution_clock::now();
    for (auto i = 0; i < COUNT; ++i) {
        preprocessor.evaluateRGBA8(rgba.data());
    }
    high_resolution_clock::time_point end = high_resolution_clock::now();
    auto elapsed = duration_cast<microseconds>(end - start).count();
    std::cout << height << "x" << width << " "<< elapsed/1000./COUNT << " ms.\n";
}

static const std::vector<DimsDef> s_dimsDef = {
        { 720, 1280},
        {1080, 1920},
        {2160, 3840},
};

//TEST_P(PerfOpenCV, PerfOpenCVTest) {
//    srand(33);
//    auto dims = GetParam();
//    uint32_t height = std::get<0>(dims);
//    uint32_t width  = std::get<1>(dims);
//    auto sz = height * width;
//    std::vector<uint8_t> rgba(sz * 4);
//
//    fill_rgba(rgba);
//
//    cv::Mat frame = cv::Mat(width, height, CV_8UC4, (unsigned*)rgba.data());
//    cv::Mat blob;
//    auto preprocessor = Preprocessor<float>(height, width);
//    high_resolution_clock::time_point start = high_resolution_clock::now();
//    for (auto i = 0; i < COUNT; ++i) {
//        cv::dnn::blobFromImage(frame, blob, 1. / 255.f, cv::Size(width, height), 0, false);
//    }
//    high_resolution_clock::time_point end = high_resolution_clock::now();
//    auto elapsed = duration_cast<microseconds>(end - start).count();
//    std::cout << height << "x" << width << " "<< elapsed/1000./COUNT << " ms.\n";
//}

//TEST(Preprocessor, CompareOpenCVwithCustom) {
//    srand(251);
//    uint32_t height = 128;
//    uint32_t width  = 128;
//    auto sz = height * width;
//    std::vector<uint8_t> rgba(sz * 4);
//    fill_rgba(rgba);
//    cv::Mat frame = cv::Mat(width, height, CV_8UC4, (unsigned*)rgba.data());
//    cv::Mat blob;
//    cv::dnn::blobFromImage(frame, blob, 1. / 255.f, cv::Size(width, height));
//    auto preprocessor = Preprocessor<float>(height, width);
//    preprocessor.evaluateRGBA8(rgba.data());
//    float* pDec = static_cast<float*>(preprocessor.getResult()->data);
//    float* pCV = (float*)(blob.data);
//    for (uint32_t i = 0; i < sz * 3; i ++) {
//        ASSERT_FLOAT_EQ(pDec[i], pCV[i]);
//    }
//}

TEST(Preprocessor, CompareFP16withFP32) {
    srand(71);
    uint32_t height = 128;
    uint32_t width  = 128;
    auto sz = height * width;
    std::vector<uint8_t> rgba(sz * 4);
    fill_rgba(rgba);

    auto preprocessorFP32 = Preprocessor<float>(height, width);
    auto preprocessorFP16 = Preprocessor<float16_t>(height, width);

    preprocessorFP32.evaluateRGBA8(rgba.data());
    preprocessorFP16.evaluateRGBA8(rgba.data());
    float* pDec32 = static_cast<float*>(preprocessorFP32.getResult()->data);
    float16_t* pDec16 = static_cast<float16_t*>(preprocessorFP16.getResult()->data);

    for (uint32_t i = 0; i < sz * 3; i ++) {
        ASSERT_NEAR(pDec32[i], float(pDec16[i]), 0.001f);
    }
}


INSTANTIATE_TEST_CASE_P(
        PerfCustomSuiteFP32,
        PerfCustom,
        ::testing::ValuesIn( s_dimsDef ));

INSTANTIATE_TEST_CASE_P(
        PerfCustomSuiteFP16,
        PerfCustomFP16,
        ::testing::ValuesIn( s_dimsDef ));

INSTANTIATE_TEST_CASE_P(
        PerfOpenCVSuite,
        PerfOpenCV,
        ::testing::ValuesIn( s_dimsDef ));


template<class T> void postprocess(T* fgr,
                                   T* pha,
                                   uint32_t height,
                                   uint32_t width,
                                   uint8_t* pixelPtr)
{
    const float scale = 1.f/255.f;
    const std::vector<T>  bgr = {T(120 * scale), T(255 * scale), T(155 * scale)};
    const T* fgr_data = fgr;
    const T* pha_data = pha;
    size_t chan_off = height * width;
    for (size_t h = 0; h < height; ++h) {
        size_t off = h * width;
        for (size_t w = 0; w < width; ++w) {
            auto fgr_r = fgr_data[0 * chan_off + off + w];
            auto fgr_g = fgr_data[1 * chan_off + off + w];
            auto fgr_b = fgr_data[2 * chan_off + off + w];
            auto alpha = pha_data[off + w];
            auto out_r = 255 * (fgr_r * alpha + bgr[0] * (1 - alpha));
            auto out_g = 255 * (fgr_g * alpha + bgr[1] * (1 - alpha));
            auto out_b = 255 * (fgr_b * alpha + bgr[2] * (1 - alpha));

            unsigned char r = (unsigned char)out_r;
            unsigned char g = (unsigned char)out_g;
            unsigned char b = (unsigned char)out_b;

            pixelPtr[3 * (off + w) + 0] = r;
            pixelPtr[3 * (off + w) + 1] = g;
            pixelPtr[3 * (off + w) + 2] = b;
        }
    }
}

template<class T>
void postprocess_tricky(
        T* pha,
        uint32_t height,
        uint32_t width,
        uint8_t* orig,
        uint8_t* pixelPtr)
{

    const std::vector<uint8_t>  bgr = {120, 255, 155};

    const T* pha_data = static_cast<T*>(pha);
    for (size_t h = 0; h < height; ++h) {
        size_t off = h * width;
        for (size_t w = 0; w < width; ++w) {
            auto alpha = pha_data[off + w];
            alpha *= 255;
            uint8_t al_v = (uint8_t)alpha;
            uint32_t out_r = (orig[4 * (off + w) + 0] * al_v + bgr[0] * (255 - al_v)) >> 8;
            uint32_t out_g = (orig[4 * (off + w) + 1] * al_v + bgr[1] * (255 - al_v)) >> 8;
            uint32_t out_b = (orig[4 * (off + w) + 2] * al_v + bgr[2] * (255 - al_v)) >> 8;
            unsigned char r = (unsigned char)out_r;
            unsigned char g = (unsigned char)out_g;
            unsigned char b = (unsigned char)out_b;
            pixelPtr[4 * (off + w) + 0] = r;
            pixelPtr[4 * (off + w) + 1] = g;
            pixelPtr[4 * (off + w) + 2] = b;
            pixelPtr[4 * (off + w) + 3] = orig[4 * (off + w) + 3];
        }
    }
}

TEST(PosprocessorTest, FP32Accuracy) {
    srand(124);
    tvm::Device cpu_ctx{kDLCPU, 0};
    int64_t height = 64;
    int64_t width  = 512;
    auto sz = height * width;
    std::vector<uint8_t> rgba_res(sz * 4);
    std::vector<uint8_t> rgba_ref(sz * 3);

    auto frm = gen_zero_data<float>({1, 3, height, width}, cpu_ctx, "float32");
    auto pha = gen_zero_data<float>({1, 1, height, width}, cpu_ctx, "float32");;
    fill_float<float>(frm);
    fill_float<float>(pha);
    auto postprocessor = Postprocessor<float>();
    postprocessor.convertToRGBA8(frm, pha, rgba_res.data());
    postprocess<float>((float*)frm->data, (float*)pha->data, height, width, rgba_ref.data());
    for (size_t i = 0; i < sz; ++i) {
        ASSERT_LE(std::fabs(rgba_res[4 * i] - rgba_ref[3 * i]), 1);
        ASSERT_LE(std::fabs(rgba_res[4 * i + 1] - rgba_ref[3 * i + 1]), 1);
        ASSERT_LE(std::fabs(rgba_res[4 * i + 2] - rgba_ref[3 * i + 2]), 1);
        ASSERT_EQ(rgba_res[4 * i + 3], ALPHA);
    }
}

TEST(PosprocessorTest, FP32TrickAccuracy) {
    srand(124);
    tvm::Device cpu_ctx{kDLCPU, 0};
    int64_t height = 64;
    int64_t width  = 512;
    auto sz = height * width;
    std::vector<uint8_t> rgba(sz * 4);
    std::vector<uint8_t> rgba_ref_res(sz * 4);
    fill_rgba(rgba);
    auto pha = gen_zero_data<float>({1, 1, height, width}, cpu_ctx, "float32");;
    fill_float<float>(pha);
    postprocess_tricky<float>((float*)pha->data, height, width, rgba.data(), rgba_ref_res.data());
    auto postprocessor = Postprocessor<float>();
    postprocessor.trickyConvertToRGBA8(pha, rgba.data());

    for (size_t i = 0; i < rgba.size(); ++i) {
        ASSERT_EQ(rgba[i], rgba_ref_res[i]);
    }
}

//TEST(PosprocessorTest, FP16TrickAccuracy) {
//    srand(124);
//    tvm::Device cpu_ctx{kDLCPU, 0};
//    int64_t height = 64;
//    int64_t width  = 512;
//    auto sz = height * width;
//    std::vector<uint8_t> rgba(sz * 4);
//    std::vector<uint8_t> rgba_ref_res(sz * 4);
//    fill_rgba(rgba);
//    auto pha = gen_zero_data<float16_t>({1, 1, height, width}, cpu_ctx, "float16");;
//    fill_float<float16_t>(pha);
//    postprocess_tricky<float16_t>((float16_t*)pha->data, height, width, rgba.data(), rgba_ref_res.data());
//    auto postprocessor = Postprocessor<float16_t>();
//    postprocessor.trickyConvertToRGBA8(pha, rgba.data());
//
//    for (size_t i = 0; i < rgba.size(); ++i) {
//        if (std::fabs(rgba[i] - rgba_ref_res[i]) > 2)
//        {
//            std::cout << i << ": " << (int)rgba[i] <<", ref " << (int)rgba_ref_res[i] << "\n";
//        }
////        ASSERT_LE(std::fabs(rgba_res[4 * i] - rgba_ref_res[4 * i]), 1);
//    }
//}

TEST(PosprocessorTest, FP16Accuracy) {
    srand(33);
    uint32_t height = 64;
    uint32_t width  = 512;
    auto sz = height * width;
    std::vector<uint8_t> rgba_res(sz * 4);
    std::vector<uint8_t> rgba_ref(sz * 3);
    tvm::Device cpu_ctx{kDLCPU, 0};
    auto frm = gen_zero_data<float16_t>({1, 3, height, width}, cpu_ctx, "float16");
    auto pha = gen_zero_data<float16_t>({1, 1, height, width}, cpu_ctx, "float16");
    fill_float<float16_t>(frm);
    fill_float<float16_t>(pha);
    auto postprocessor = Postprocessor<float16_t>();
    postprocessor.convertToRGBA8(frm, pha, rgba_res.data());
    postprocess<float16_t>((float16_t*)frm->data, (float16_t*)pha->data, height, width, rgba_ref.data());
    for (size_t i = 0; i < sz; ++i) {
        ASSERT_LE(std::fabs(rgba_res[4 * i] - rgba_ref[3 * i]), 1);
        ASSERT_LE(std::fabs(rgba_res[4 * i + 1] - rgba_ref[3 * i + 1]), 1);
        ASSERT_LE(std::fabs(rgba_res[4 * i + 2] - rgba_ref[3 * i + 2]), 1);
        ASSERT_EQ(rgba_res[4 * i + 3], ALPHA);
    }
}

TEST_P(PerfPostprocessorFP32, PerfPostprocessor) {
    srand(124);
    tvm::Device cpu_ctx{kDLCPU, 0};
    auto dims = GetParam();
    uint32_t height = std::get<0>(dims);
    uint32_t width  = std::get<1>(dims);
    auto sz = height * width;
    std::vector<uint8_t> rgba(sz * 4);
    auto frm = gen_zero_data<float32_t>({1, 3, height, width}, cpu_ctx, "float32");
    auto pha = gen_zero_data<float32_t>({1, 1, height, width}, cpu_ctx, "float32");;
    fill_float<float32_t>(frm);
    fill_float<float32_t>(pha);

    auto postprocessor = Postprocessor<float>();
    high_resolution_clock::time_point start = high_resolution_clock::now();
    for (auto i = 0; i < COUNT; ++i) {
        postprocessor.convertToRGBA8(frm, pha, rgba.data());
    }
    high_resolution_clock::time_point end = high_resolution_clock::now();
    auto elapsed = duration_cast<microseconds>(end - start).count();
    std::cout << height << "x" << width << " "<< elapsed/1000./COUNT << " ms.\n";
}

TEST_P(PerfTrickyPostprocessorFP32, PerfPostprocessor) {
    srand(124);
    tvm::Device cpu_ctx{kDLCPU, 0};
    auto dims = GetParam();
    uint32_t height = std::get<0>(dims);
    uint32_t width  = std::get<1>(dims);
    auto sz = height * width;
    std::vector<uint8_t> rgba(sz * 4);
    auto pha = gen_zero_data<float32_t>({1, 1, height, width}, cpu_ctx, "float32");;
    fill_float<float32_t>(pha);

    auto postprocessor = Postprocessor<float>();
    high_resolution_clock::time_point start = high_resolution_clock::now();
    for (auto i = 0; i < COUNT; ++i) {
        postprocessor.trickyConvertToRGBA8(pha, rgba.data());
    }
    high_resolution_clock::time_point end = high_resolution_clock::now();
    auto elapsed = duration_cast<microseconds>(end - start).count();
    std::cout << height << "x" << width << " "<< elapsed/1000./COUNT << " ms.\n";
}

TEST_P(PerfPostprocessorFP16, PerfPostprocessor) {
    srand(124);
    tvm::Device cpu_ctx{kDLCPU, 0};
    auto dims = GetParam();
    uint32_t height = std::get<0>(dims);
    uint32_t width  = std::get<1>(dims);
    auto sz = height * width;
    std::vector<uint8_t> rgba(sz * 4);
    auto frm = gen_zero_data<float16_t>({1, 3, height, width}, cpu_ctx, "float16");
    auto pha = gen_zero_data<float16_t>({1, 1, height, width}, cpu_ctx, "float16");
    fill_float<float16_t>(frm);
    fill_float<float16_t>(pha);


    auto postprocessor = Postprocessor<float16_t>();
    high_resolution_clock::time_point start = high_resolution_clock::now();
    for (auto i = 0; i < COUNT; ++i) {
        postprocessor.convertToRGBA8(frm, pha, rgba.data());
    }
    high_resolution_clock::time_point end = high_resolution_clock::now();
    auto elapsed = duration_cast<microseconds>(end - start).count();
    std::cout << height << "x" << width << " "<< elapsed/1000./COUNT << " ms.\n";
}

//TEST_P(PerfTrickyPostprocessorFP16, PerfPostprocessor) {
//    srand(1240);
//    tvm::Device cpu_ctx{kDLCPU, 0};
//    auto dims = GetParam();
//    uint32_t height = std::get<0>(dims);
//    uint32_t width  = std::get<1>(dims);
//    auto sz = height * width;
//    std::vector<uint8_t> rgba(sz * 4);
//    auto pha = gen_zero_data<float16_t>({1, 1, height, width}, cpu_ctx, "float16");;
//    fill_float<float16_t>(pha);
//
//    auto postprocessor = Postprocessor<float16_t>();
//    high_resolution_clock::time_point start = high_resolution_clock::now();
//    for (auto i = 0; i < COUNT; ++i) {
//        postprocessor.trickyConvertToRGBA8(pha, rgba.data());
//    }
//    high_resolution_clock::time_point end = high_resolution_clock::now();
//    auto elapsed = duration_cast<microseconds>(end - start).count();
//    std::cout << height << "x" << width << " "<< elapsed/1000./COUNT << " ms.\n";
//}

INSTANTIATE_TEST_CASE_P(
        PerfPostprocessorSuiteFP32,
        PerfPostprocessorFP32,
        ::testing::ValuesIn( s_dimsDef ));

INSTANTIATE_TEST_CASE_P(
        PerfTrickyPostprocessorSuiteFP32,
        PerfTrickyPostprocessorFP32,
        ::testing::ValuesIn( s_dimsDef ));

INSTANTIATE_TEST_CASE_P(
        PerfPostprocessorSuiteFP16,
        PerfPostprocessorFP16,
        ::testing::ValuesIn( s_dimsDef ));

//INSTANTIATE_TEST_CASE_P(
//        PerfTrickyPostprocessorSuiteFP16,
//        PerfTrickyPostprocessorFP16,
//        ::testing::ValuesIn( s_dimsDef ));
