/*!
 * \brief Header file for code to load and run a TVM module.
 * \file model.h
 */
#ifndef MODEL_H_
#define MODEL_H_

#include <iostream>
#include <map>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <cstdio>
#include  <mutex>
#if defined __aarch64__
#include <arm_neon.h>
#include "helper_utils.h"
#endif //__aarch64__

#include <dlpack/dlpack.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/vm/vm.h>
#include <dlpack/dlpack.h>
#include <tvm/runtime/container/adt.h>

#include "inputs.h"
/*!
 * \brief Executable
 *  This class is an interface for loading of exported relay model libs.
 *  This separation is required for weights sharing between different execution instances.
 */

using ExeModule = tvm::runtime::ObjectPtr<tvm::runtime::vm::Executable>;
using Name2Ind = std::map<std::string, uint32_t>;

constexpr size_t RNN_OUTPUTS_NUM = 4;

// inline void loadNDArray(const std::string& nm,
//                         const std::string& pth,
//                         tvm::runtime::NDArray arr,
//                         tvm::runtime::PackedFunc set_input)
// {
//     auto shape = arr.Shape();
//     size_t sz = 1;
//     for (size_t i = 0; i < shape.size(); ++i) {
//         sz *= shape[i];
//     }
//     size_t bytes = arr.DataType().bytes() * sz;
//     auto f = fopen((pth + nm).c_str(), "rb");
//     if (f) {

//         auto reads = fread((char*)arr->data, 1, bytes, f);
//         if (reads != bytes) {
//             std::cout << "ERROR: read " << reads << ", but required " << bytes << " bytes from " << nm << " file.\n";
//         }
//         fclose(f);
//     } else {
//         std::cout << " File: " << (pth + nm).c_str() << " was not found.\n";

//     }
//     if (set_input != nullptr) {
//         set_input(nm, arr);
//     }
// }

// inline void loadData(
//               std::vector<tvm::runtime::NDArray>& outData,
//               const DLDevice& ctx,
//               tvm::runtime::PackedFunc set_input,
//               const std::string& pth) {
//   if constexpr(USE_GE == true){
//     outData.resize(s_dataMap.size());
//     size_t ind = 0;
//     size_t idx = 0;
//     for (auto item : s_dataMap) {
//       const auto& el = item;
//       std::vector<int64_t> shp;
//       for (size_t i = 0; i < el.shape.size(); ++i) {
//         shp.push_back(el.shape[i]);
//       }
//       if (el.shape.size() < 3) {
//         size_t num = 3 - el.shape.size();
//         for (size_t j = 0; j < num; ++j) {
//           shp.push_back(1);
//         }
//       }
//       outData[ind] = tvm::runtime::NDArray::Empty(tvm::runtime::ShapeTuple(shp.begin(), shp.end()),
//                                                   tvm::runtime::String2DLDataType(el.dtype), {kDLCPU, 0});
//       loadNDArray(el.name, pth, outData[ind], set_input);
//       ind++;
//     }
//   }
// }

template <typename T>
tvm::runtime::NDArray tvm_ndarray_from_vec(
        std::vector<T> data,
        std::vector<int64_t> shape,
        tvm::Device ctx,
        std::string numpy_dtype,
        int64_t num_elements
)
{
  // Initialize the NDArray
  DLDataType dl_data_type = tvm::runtime::String2DLDataType(numpy_dtype);
  tvm::runtime::NDArray input_arr = tvm::runtime::NDArray::Empty(
          shape,
          dl_data_type,
          ctx
  );

  // Copy the generated data to the NDArray
  int64_t num_bytes = num_elements * sizeof(T);
  input_arr.CopyFromBytes(data.data(), num_bytes);
  return input_arr;
}

template <typename T>
tvm::runtime::NDArray gen_zero_data(
        std::vector<int64_t> shape,
        tvm::Device ctx,
        std::string numpy_dtype
)
{
  std::vector<tvm::runtime::NDArray> input_arrs;

  int64_t num_elements = 1;
  for (int64_t s : shape) {
    num_elements *= s;
  }

  std::vector<T> data(num_elements);

  tvm::runtime::NDArray input_arr = tvm_ndarray_from_vec<T>(
          data,
          shape,
          ctx,
          numpy_dtype,
          num_elements
  );
  return input_arr;
}

static inline
void init_output_objects(std::vector<tvm::runtime::NDArray>& outputs)
{
  tvm::Device cpu_ctx{kDLCPU, 0};
  auto pha_shape = s_model_inputs[0].shape;
  pha_shape[1] = 1;

  outputs.resize(2);
  DLDataType dl_data_type = tvm::runtime::String2DLDataType(s_model_inputs[0].dtype);

  outputs[0] = tvm::runtime::NDArray::Empty(s_model_inputs[0].shape, dl_data_type, cpu_ctx);
  outputs[1] = tvm::runtime::NDArray::Empty(pha_shape, dl_data_type, cpu_ctx);
}

class Executable {
public:
    Executable() {}
    void init(
            const std::string& lib_path,
            const std::string& consts_path,
            const std::string& vm_exec_code_path,
            const tvm::Device& context
    )
    {
        context_ = context;
        std::cout << lib_path << "\n";
        tvm::runtime::Module lib = tvm::runtime::Module::LoadFromFile(lib_path);
        if constexpr (USE_GE == true) {
            // loading of graph executor
            exec_mod_ = lib.GetFunction("default")(context_);
            if (exec_mod_.get() == nullptr) {
                std::cout << "ERROR: executable was not created " << vm_exec_code_path << "\n";
                return;
            }
            // tvm::runtime::PackedFunc share_weights_func;
            // auto call = exec_mod_.GetFunction("share_params");
            // std::cout << " share_parmas "<< call.get() << "\n";
            // if (call.get() != nullptr) {
            //     // tvm::runtime::Module exe = module.get_tvm_module();
            //     call(exec_mod_, s_params_to_share);
            // }

            // std::cout << "executable created\n";
            // auto pfr = tvm::runtime::Registry::Get("tvm.graph_executor.create");
            // ICHECK(mod.defined()) << "Module must be defined";
            // tvm::runtime::Module run_mod =
            // (*pfr)(json, mod, static_cast<int>(dev.device_type), dev.device_id);
  
        }else {
            std::cout << "vm_exec_code_path " << vm_exec_code_path << "\n";
            std::ifstream code(vm_exec_code_path, std::ios::binary);
            std::stringstream ss;
            ss << code.rdbuf();

            exec_mod_ = tvm::runtime::vm::Executable::Load(ss.str(), lib);
            // std::cout << "executable loaded\n";
            if (exec_mod_.get() == nullptr) {
                return;
            }
            const tvm::runtime::vm::Executable* tmp = exec_mod_.as<tvm::runtime::vm::Executable>();
            exec_ = tvm::runtime::GetObjectPtr<tvm::runtime::vm::Executable>(const_cast<tvm::runtime::vm::Executable*>(tmp));
            exec_->LoadLateBoundConstantsFromFile(consts_path);
            std::cout << "consts  loaded\n";
            auto params_num = exec_->GetFunctionArity("main");
            for (size_t i = 0; i < params_num; ++i) {
                std::string iname = exec_->GetFunctionParameterName("main", i);
                name_to_index_[iname] = i;
            }

        }
        std::cout << "executable loaded\n";
    }
    ExeModule get_module() const {
        return exec_;
    }
    const Name2Ind& get_names_mappings() const {
        return name_to_index_;
    }
    tvm::runtime::Module get_tvm_module() {
        return exec_mod_;
    }
    void share_weights(Executable& root) {
    }
protected:
    /*! \brief The loaded module lib*/
    tvm::runtime::Module exec_mod_;
    /*! \brief The vm executable */
    ExeModule exec_;
    /*! \brief Input parameter name mapping to input index */
    Name2Ind name_to_index_;
    tvm::Device context_;
};

static inline
void init_RNN_objects(std::vector<tvm::runtime::NDArray>& outputs)
{
  outputs.resize(RNN_OUTPUTS_NUM);
  tvm::Device cpu_ctx{kDLCPU, 0};

  for (size_t i = 0; i < outputs.size(); ++i) {
      outputs[i] = gen_zero_data<RVMInputType>(s_model_inputs[i + 1].shape, cpu_ctx, s_model_inputs[i + 1].dtype);
  }
}

static inline
void reset_RNN_objects(std::vector<tvm::runtime::NDArray>& outputs)
{
  outputs.resize(RNN_OUTPUTS_NUM);
  tvm::Device cpu_ctx{kDLCPU, 0};

  for (size_t i = 0; i < outputs.size(); ++i) {
    size_t data_size = 1;
    auto shape = outputs[i].Shape();
    for (size_t j = 0; j < shape.size(); ++j) {
        data_size *= shape[j];
    }
    size_t bytes = 4;
    if (outputs[i].DataType().is_float16()) {
        bytes = 2;
    }
    if (outputs[i]->data){
        std::memset(outputs[i]->data, 0, data_size * bytes);
    }
  }
}

#if defined __aarch64__
static inline
void copy_FP16_toFP32(const tvm::runtime::NDArray& from,
                      tvm::runtime::NDArray& to)
{
  if (from->data == nullptr ||
      to->data == nullptr) {
      return;
  }
  auto shape_from = from.Shape();
  auto shape_to = to.Shape();
  if (shape_from.size() != shape_to.size()) {
      return;
  }
  size_t from_size = 1;
  size_t to_size = 1;
  for (size_t j = 0; j < shape_from.size(); ++j) {
      from_size *= shape_from[j];
      to_size *= shape_to[j];
  }
  if (from_size != to_size) {
      return;
  }
  const float16_t* pFrom = static_cast<const float16_t*>(from->data);
  float32_t* pTo = static_cast<float32_t*>(to->data);
  size_t j = 0;
  convertFP16toFP32(pFrom, pTo, from_size);
}

#endif // __aarch64__

/*!
 * \brief Model
 *  This class is an interface for setting inputs, and running
 *  of exported relay model libs.
 */
template <bool useGE> class Model;

template <>
class Model<false> {
public:
    /*! \brief The set_input function for the vm */
    tvm::runtime::PackedFunc set_input_func;
    /*! \brief The get_output function for the vm */
    tvm::runtime::PackedFunc get_output_func;
    /*! \brief The run function for the vm */
    tvm::runtime::PackedFunc run_func;

    /*! \brief The VM initialized from libpath */
    tvm::runtime::vm::VirtualMachine vm_;
    std::vector<tvm::runtime::NDArray> outputs_;
    size_t ind_ = 0;
    size_t rec0_ind_ = 1;
    size_t rec1_ind_ = 2;
    size_t rec2_ind_ = 3;
    size_t rec3_ind_ = 4;

    /*!
     * \brief constructor.
     * \param lib_path string indicating path to the .so lib containing the TVM runtime module.
     * \param consts_path string indicating path to the consts file containing the model's parameters.
     * \param vm_exec_code_path string indicating path to the .ro file containing the vm serialization.
     * \param dl_device_type The device type of the run environment -- should be kDLCPU or kDLGPU.
     */
    Model(Executable& module)
    {
        ExeModule exe = module.get_module();
        switch (DL_DEVICE_TYPE)
        {
            case kDLCPU:
                /* code */
            {
                vm_.LoadExecutable(exe);

                // Initialize the VM for the specified device. If the device is not a CPU,
                // We'll need to add a CPU context to drive it.
                int arity;
                arity = 3;
                // Specify how to allocate memory for the target devices.
                uint64_t alloc_type = uint64_t(tvm::runtime::vm::AllocatorType::kPooled);
                // Always use the first device of the specified type.
                uint64_t device_id = 0;
                // Create a variable length input to the packed function.
                std::vector<TVMValue> init_vals(arity);
                std::vector<int> codes(arity);
                tvm::runtime::TVMArgsSetter setter(init_vals.data(), codes.data());
                // Set up the main device context.
                setter(0, (uint64_t (DL_DEVICE_TYPE)));
                setter(1, device_id);
                setter(2, alloc_type);
                // Also initialize a CPU device context.
                if constexpr (DL_DEVICE_TYPE != kDLCPU) {
                    setter(3, (uint64_t (kDLCPU)));
                    setter(4, device_id);
                    setter(5, alloc_type);
                }
                tvm::runtime::TVMRetValue rv;
                // Call the packed func with the init arguments.
                vm_.GetFunction("init", nullptr).CallPacked(tvm::runtime::TVMArgs(init_vals.data(), codes.data(), arity), &rv);

                set_input_func = vm_.GetFunction("set_input", nullptr);
                get_output_func = vm_.GetFunction("get_output", nullptr);
                run_func = vm_.GetFunction("invoke", nullptr);
                if (s_model_inputs.size() == 5){
                    init_RNN_objects(outputs_);
                }
                auto names_mappings = module.get_names_mappings();

                if (s_model_inputs.size() == 5){
                    ind_ = names_mappings[s_model_inputs[0].name] + 1;
                    rec0_ind_ = names_mappings[s_model_inputs[1].name] + 1;
                    rec1_ind_ = names_mappings[s_model_inputs[2].name] + 1;
                    rec2_ind_ = names_mappings[s_model_inputs[3].name] + 1;
                    rec3_ind_ = names_mappings[s_model_inputs[4].name] + 1;
                } else {
                    // just resnet50
                    ind_ = names_mappings[s_model_inputs[0].name] + 1;
                }
            }
                break;
            default:
                break;
        }
    }

    /**
     * Run model inference with the inputs given by `input_vec`. Pre-loaded inputs
     * will be used in the model if input_vec is empty.
     *
     * @param input_vec a vector containing input NDArrays. Defaults to empty.
     * @return vec of NDArray -- the outputs from running the model
     */
    void run(const tvm::runtime::NDArray& input_image,
             tvm::runtime::NDArray& frame,
             tvm::runtime::NDArray& alpha)
    {

        // arity is num of inputs + 1, because first argument to the set_input_func
        // is the name of the function that should take those inputs.
        size_t arity = s_model_inputs.size() + 1;
        std::vector<TVMValue> values(arity);
        std::vector<int> codes(arity);
        tvm::runtime::TVMArgsSetter setter(values.data(), codes.data());
        setter(0, "main");
        setter(ind_, input_image);
        if (s_model_inputs.size() == 5){
            setter(rec0_ind_, outputs_[0]);
            setter(rec1_ind_, outputs_[1]);
            setter(rec2_ind_, outputs_[2]);
            setter(rec3_ind_, outputs_[3]);
        }
        tvm::runtime::TVMRetValue rv;
        set_input_func.CallPacked(tvm::runtime::TVMArgs(values.data(), codes.data(), arity), &rv);

        tvm::runtime::ObjectRef out = run_func("main");
        TVMSynchronize((int)DL_DEVICE_TYPE, 0, nullptr);
        if (!outputs_.empty()) {
            if (out.as<tvm::runtime::ADTObj>()) {
                auto adt = tvm::Downcast<tvm::runtime::ADT>(out);
                tvm::runtime::NDArray arr = tvm::Downcast<tvm::runtime::NDArray>(adt[0]);
                // ToDo: implement zero copy

                arr.CopyTo(frame);
                arr = tvm::Downcast<tvm::runtime::NDArray>(adt[1]);
                arr.CopyTo(alpha);
                for (size_t i = 2; i < adt.size(); ++i) {
                    arr = tvm::Downcast<tvm::runtime::NDArray>(adt[i]);
#if defined __aarch64__
                    if (arr.DataType().is_float16()) {
                        copy_FP16_toFP32(arr, outputs_[i - 2]);
                    } else
#endif // __aarch64__
                    {
                        arr.CopyTo(outputs_[i - 2]);
                    }
                }
            } else {
                tvm::runtime::NDArray arr = tvm::Downcast<tvm::runtime::NDArray>(out);
                arr.CopyTo(frame);
            }
        }
    }
    void cleanRNNState() {
        reset_RNN_objects(outputs_);
    }
};

template <>
class Model<true> {
public:
    /*! \brief The set_input function for the vm */
    tvm::runtime::PackedFunc set_input_func;
    /*! \brief The get_output function for the vm */
    tvm::runtime::PackedFunc get_output_func;
    /*! \brief The run function for the vm */
    tvm::runtime::PackedFunc run_func;
    // tvm::runtime::PackedFunc set_input_zero_copy_func;
#if defined __aarch64__
    // static std::vector<tvm::runtime::NDArray> outputsFP16_;
    // static std::once_flag initialized_FP16output;
    std::vector<tvm::runtime::NDArray> outputsFP16_;

#endif
    // static std::once_flag initialized_output;
    // static std::vector<tvm::runtime::NDArray> outputs_;

    std::vector<tvm::runtime::NDArray> outputs_;

    Model(Executable& module)
    {
        tvm::runtime::Module exe = module.get_tvm_module();
        set_input_func = exe.GetFunction("set_input");
        get_output_func = exe.GetFunction("get_output");
        run_func = exe.GetFunction("run");
        // set_input_zero_copy_func = exe.GetFunction("set_input_zero_copy");
        // share_params_func = exe.GetFunction("share_params");
        // if (s_model_inputs.size() == 5){
        //     std::call_once(initialized_output, init_RNN_objects, outputs_);
        // }
        if (s_model_inputs.size() == 5){
            init_RNN_objects(outputs_);
        }

#if defined __aarch64__
        if constexpr (USE_FP16 == true) {
            if (s_model_inputs.size() == 5){
                // std::call_once(initialized_FP16output, [&] {
                    // workaround. Hope will be fixed soon
                    outputsFP16_.resize(RNN_OUTPUTS_NUM);
                    tvm::Device cpu_ctx{kDLCPU, 0};

                    for (size_t i = 0; i < outputsFP16_.size(); ++i) {
                        outputsFP16_[i] = gen_zero_data<float16_t>(s_model_inputs[i + 1].shape, cpu_ctx, "float16");
                    }
                // });
            }
        }
#endif
    }
    ~Model() = default;

    void set_inputs(const tvm::runtime::NDArray& input_image) {
        if (s_model_inputs.size() == 5){
            if (set_input_func.get() != nullptr) {
                set_input_func(s_model_inputs[0].name, input_image);
                // set_input_zero_copy_func(s_model_inputs[0].name, input_image);
                for (size_t i = 0; i < outputs_.size(); ++i) {
                    set_input_func(s_model_inputs[i + 1].name, outputs_[i]);
                    // set_input_zero_copy_func(s_model_inputs[i + 1].name, outputs_[i]);
                }
            }
        }
    }

    void run()
    {
        if (run_func.get() != nullptr) {
            run_func();
        }
    }
    void synchronize() {
        if ((outputs_.size() == 4) && get_output_func.get() != nullptr) {
#if defined __aarch64__
            if constexpr (USE_FP16 == false) {
                for (size_t i = 0; i < outputs_.size(); ++i) {
                    get_output_func(2 + i, outputs_[i]);
                }
            } else {
                for (size_t i = 0; i < outputsFP16_.size(); ++i) {
                    get_output_func(2 + i, outputsFP16_[i]);
//                    copy_FP16_toFP32(outputsFP16_[i], outputs_[i]);
                }
            }
#else
        for (size_t i = 0; i < outputs_.size(); ++i) {
          get_output_func(2 + i, outputs_[i]);
        }
#endif
        TVMSynchronize(DL_DEVICE_TYPE, 0, nullptr);

        for (size_t i = 0; i < outputsFP16_.size(); ++i) {
            copy_FP16_toFP32(outputsFP16_[i], outputs_[i]);
        }

        }
    }
    void get_outuputs(tvm::runtime::NDArray& frame,
             tvm::runtime::NDArray& alpha)
    {
        if ((outputs_.size() == 4) && get_output_func.get() != nullptr) {
            get_output_func(0, frame);
            get_output_func(1, alpha);
        }
    }

    void share_params(Executable& module) {
        // if (share_params_func.get() != nullptr) {
        //     tvm::runtime::Module exe = module.get_tvm_module();
        //     share_params_func(exe, s_params_to_share);
        // }
    }
    void cleanRNNState() {
      reset_RNN_objects(outputs_);
    }
};

#endif
