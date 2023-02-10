/*
 *
 */

#include <jni.h>
#include <string>
#include <memory>
#include "inputs.h"
#include "model.h"
#include <android/bitmap.h>
#include <android/log.h>
#include "Preprocessor.h"
#include "Postprocessor.h"
#include "RVMExecutor.h"
#include <media/NdkImageReader.h>
#include <chrono>

using namespace std::chrono;

#define APPNAME "JNILog"
#define LOGGER(...) { \
    __android_log_print(ANDROID_LOG_VERBOSE, APPNAME, __VA_ARGS__); \
}

static std::string s_tmp_dir = "";
static std::shared_ptr<Preprocessor<RVMInputType>> getPreprocessor()
{
    static std::shared_ptr<Preprocessor<RVMInputType>> preprocessor =
                    std::make_shared<Preprocessor<RVMInputType>>();
    return preprocessor;
}

static std::shared_ptr<RVMExecutor> getExecutor()
{
    static std::shared_ptr<RVMExecutor> executor =
            std::make_shared<RVMExecutor>(s_tmp_dir);
    return executor;
}

static std::shared_ptr<Postprocessor<RVMInputType>> getPostprocessor()
{
    static std::shared_ptr<Postprocessor<RVMInputType>> postprocessor =
            std::make_shared<Postprocessor<RVMInputType>>();
    return postprocessor;
}

std::string ConvertJString(JNIEnv* env, jstring str)
{
    if ( !str ) std::string();

    const jsize len = env->GetStringUTFLength(str);
    const char* strChars = env->GetStringUTFChars(str, (jboolean *)0);

    std::string Result(strChars, len);

    env->ReleaseStringUTFChars(str, strChars);

    return Result;
}

extern "C"
JNIEXPORT jint JNICALL
Java_com_example_android_RVM_TVM_CameraActivity_initRVM(JNIEnv *env, jobject thiz, jstring path) {
    s_tmp_dir = ConvertJString( env, path );
    LOGGER("path is  = %s", s_tmp_dir.c_str());
    return 45;
}

extern "C"
JNIEXPORT jint JNICALL
Java_com_example_android_RVM_TVM_CameraActivity_releaseRVM(JNIEnv *env, jobject thiz) {
    return 0;
}

extern "C"
JNIEXPORT jint JNICALL
Java_com_example_android_RVM_TVM_CameraActivity_updateBitmap(JNIEnv *env, jobject thiz,
                                                             jobject image, jobject proxy) {
//    {
//        jclass cls = env->GetObjectClass(proxy);
//
//        // First get the class object
//        jmethodID mid = env->GetMethodID(cls, "getClass", "()Ljava/lang/Class;");
//        jobject clsObj = env->CallObjectMethod(proxy, mid);
//
//        // Now get the class object's class descriptor
//        cls = env->GetObjectClass(clsObj);
//
//        // Find the getName() method on the class object
//        mid = env->GetMethodID(cls, "getName", "()Ljava/lang/String;");
//
//        // Call the getName() to get a jstring object back
//        jstring strObj = (jstring)env->CallObjectMethod(clsObj, mid);
//
//        // Now get the c string from the java jstring object
//        const char* str = env->GetStringUTFChars(strObj, NULL);
//
//        // Print the class name
//        LOGGER("\nCalling class is: %s\n", str);
//
//        // Release the memory pinned char array
//        env->ReleaseStringUTFChars(strObj, str);
//    }
    AndroidBitmapInfo  info;
    auto res = AndroidBitmap_getInfo(env, image, &info);
    if (res != ANDROID_BITMAP_RESULT_SUCCESS) {
        return -1;
    }
    if (info.format == ANDROID_BITMAP_FORMAT_RGBA_8888) {
        if (info.height != s_model_inputs[0].shape[2] ||  info.width != s_model_inputs[0].shape[3]) {
            LOGGER("Incorrect size w = %d, h = %d, but required w is %d, h is %d ",\
                   info.width, info.height,
                   (int)s_model_inputs[0].shape[2],
                   (int)s_model_inputs[0].shape[3]);
            return -1;
        }
        void* pData = nullptr;
        res = AndroidBitmap_lockPixels(env, image, &pData);
        if (res != ANDROID_BITMAP_RESULT_SUCCESS) {
            AndroidBitmap_unlockPixels(env, image);
        }
        {
            auto preprocessor = getPreprocessor();
            auto executor = getExecutor();
            auto postprocessor = getPostprocessor();
            {
//                LocalTimer("preprocessor");
                high_resolution_clock::time_point start = high_resolution_clock::now();
                if (preprocessor && pData) {
                    res = preprocessor->evaluateRGBA8(pData);
                }
                high_resolution_clock::time_point end = high_resolution_clock::now();
                auto real_duration_sec = duration_cast<microseconds>(end - start).count();
                LOGGER("preprocessor time: %f ms.", real_duration_sec/1000.f);
            }
            if (res != 0) {
                AndroidBitmap_unlockPixels(env, image);
                return res;
            }
            if (executor) {
//                LocalTimer("Inference");
                high_resolution_clock::time_point start = high_resolution_clock::now();
//                model->set_inputs(input_frame);
//                model->run();
//                model->get_outuputs(outputs[0], outputs[1]);
//                model->synchronize();
                executor->set_input(preprocessor->getResult());
                res = executor->inference();
                executor->get_results();
                executor->sync();
                high_resolution_clock::time_point end = high_resolution_clock::now();
                auto real_duration_sec = duration_cast<microseconds>(end - start).count();
                LOGGER("Inference time: %f ms.", real_duration_sec/1000.f);

            }
            if (res != 0) {
                AndroidBitmap_unlockPixels(env, image);
                return res;
            }

            if (postprocessor) {
                high_resolution_clock::time_point start = high_resolution_clock::now();
                res = postprocessor->convertToRGBA8(executor->getFrame(),
                                                    executor->getAlpha(),
                                                    (uint8_t *) pData);
                high_resolution_clock::time_point end = high_resolution_clock::now();
                auto real_duration_sec = duration_cast<microseconds>(end - start).count();
                LOGGER("Postprocessor time: %f ms. %x", real_duration_sec/1000.f, pData);

            }
        }
        AndroidBitmap_unlockPixels(env, image);
    } else {
        LOGGER("Unsupported format  %d", info.format);
    }
    return res;
}
extern "C"
JNIEXPORT void JNICALL
Java_com_example_android_RVM_TVM_CameraActivity_clearRNNState(JNIEnv *env, jobject thiz) {
    // TODO: implement clearRNNState()
    auto executor = getExecutor();
    if (executor) {
        executor->clearRNNState();
    }
}

extern "C"
JNIEXPORT jbyteArray JNICALL
Java_com_example_android_RVM_TVM_CameraActivity_getLibraryName(JNIEnv *env, jobject thiz) {
    jbyteArray array = env->NewByteArray(s_lib_name.length());
    env->SetByteArrayRegion(array,0,s_lib_name.length(),(jbyte*)s_lib_name.c_str());
    return array;
}
