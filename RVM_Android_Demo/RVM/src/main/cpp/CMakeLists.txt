
# For more information about using CMake with Android Studio, read the
# documentation: https://d.android.com/studio/projects/add-native-code.html

# Sets the minimum version of CMake required to build the native library.

cmake_minimum_required(VERSION 3.18.1)

set(CMAKE_BUILD_TYPE Release)
# Declares and names the project.

project("tvm_rvm")

set(TARGET_NAME tvm_rvm)
set (PROJECT_DIR ${CMAKE_CURRENT_SOURCE_DIR})

#set(OpenCV_DIR "opencv")
#find_package( OpenCV REQUIRED )

#set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_DIR})

set (LOGGING_HEADER "\"\<tvm/runtime/logging.h\>\"")
set (TVM_HOME $ENV{TVM_HOME})

#set (OPENCL_SDK_PATH $ENV{OPENCL_SDK_PATH})
#message("OPENCL_SDK_PATH ${OPENCL_SDK_PATH}")
#set(OpenCL_INCLUDE_DIRS ${OPENCL_SDK_PATH}/include)
#set(OpenCL_LIBRARIES ${OPENCL_SDK_PATH}/lib64/libOpenCL.so)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=armv8.2-a+fp16+dotprod  -DDMLC_USE_LOGGING_LIBRARY=${LOGGING_HEADER} -DTVM_USE_LIBBACKTRACE=0")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -march=armv8.2-a+fp16+dotprod  -DDMLC_USE_LOGGING_LIBRARY=${LOGGING_HEADER} -DTVM_USE_LIBBACKTRACE=0")

# Creates and names a library, sets it as either STATIC
# or SHARED, and provides the relative paths to its source code.
# You can define multiple libraries, and CMake builds them for you.
# Gradle automatically packages shared libraries with your APK.
include_directories(
#        ${OpenCV_INCLUDE_DIRS}
#        ${OpenCL_INCLUDE_DIRS}
        ${TVM_HOME}/3rdparty/dlpack/include
        ${TVM_HOME}/3rdparty/dmlc-core/include
        ${TVM_HOME}/include
        ${PROJECT_DIR}
        )

add_library(
    ${TARGET_NAME}

    SHARED

    RVMExecutor.cpp
    tvm_rvm.cpp
)

# Searches for a specified prebuilt library and stores the path as a
# variable. Because CMake includes system libraries in the search path by
# default, you only need to specify the name of the public NDK library
# you want to add. CMake verifies that the library exists before
# completing its build.

find_library( # Sets the name of the path variable.
    log-lib

    # Specifies the name of the NDK library that
    # you want CMake to locate.
    log )

add_library(tvm_runtime SHARED IMPORTED) # or STATIC instead of SHARED
set_target_properties(tvm_runtime PROPERTIES
        IMPORTED_LOCATION
        "${CMAKE_SOURCE_DIR}/jniLibs/${CMAKE_ANDROID_ARCH_ABI}/libtvm_runtime.so"
        )

add_library(cpp_lib SHARED IMPORTED) # or STATIC instead of SHARED
set_target_properties(cpp_lib PROPERTIES
        IMPORTED_LOCATION
        "${CMAKE_SOURCE_DIR}/jniLibs/${CMAKE_ANDROID_ARCH_ABI}/libc++_shared.so"
        )

#add_library(opencl SHARED IMPORTED) # or STATIC instead of SHARED
#set_target_properties(opencl PROPERTIES
#        IMPORTED_LOCATION
#        "${OpenCL_LIBRARIES}"
#        )

# Specifies libraries CMake should link to your target library. You
# can link multiple libraries, such as libraries you define in this
# build script, prebuilt third-party libraries, or system libraries.

target_link_libraries( # Specifies the target library.
    ${TARGET_NAME}
    tvm_runtime
    # Links the target library to the log library
    # included in the NDK.
    ${log-lib}
    jnigraphics
#    ${OpenCV_LIBS}
    cpp_lib
#    opencl
    -fuse-ld=gold
)