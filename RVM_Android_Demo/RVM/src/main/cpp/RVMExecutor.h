//
//

#ifndef ANDROIDDEMO_RVMEXECUTOR_H
#define ANDROIDDEMO_RVMEXECUTOR_H

#include "model.h"

class RVMExecutor {
public:
    RVMExecutor(const std::string& libPath);
    RVMExecutor();
    int inference() {
        if (completeInitialization_ == false || model_ == nullptr ) {
            return -1;
        }
        // model_->run(input, outputs_[0], outputs_[1]);

        model_->run();
        return 0;
    };
    void sync(){
        if (completeInitialization_ == false || model_ == nullptr ) {
            return;
        }
        model_->synchronize();
    }

    void get_results() {
        if (completeInitialization_ == false || model_ == nullptr ) {
            return;
        }
        model_->get_outuputs(outputs_[0], outputs_[1]);
    }

    void set_input(const tvm::runtime::NDArray& input) {
        if (completeInitialization_ == false || model_ == nullptr ) {
            return;
        }
        model_->set_inputs(input);
    }

    const tvm::runtime::NDArray& getFrame() {
        return outputs_[0];
    }

    const tvm::runtime::NDArray& getAlpha() {
        return outputs_[1];
    }
    void clearRNNState() {
        if (model_) {
            model_->cleanRNNState();
        }
    }
    virtual ~RVMExecutor();
protected:
    Executable executable_;
    std::shared_ptr<Model<USE_GE>> model_;
    std::vector<tvm::runtime::NDArray> outputs_;
    bool completeInitialization_ = false;
};
#endif //ANDROIDDEMO_RVMEXECUTOR_H
