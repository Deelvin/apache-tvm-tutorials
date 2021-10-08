import os

import torch
import torchvision

import tvm
from tvm.contrib import graph_runtime
from tvm.contrib.download import download_testdata

from tvm.contrib import xcode
from tvm import relay, auto_scheduler

import cv2
import numpy as np

from prettytable import PrettyTable


def preprocess(img_data):
    mean_vec = np.array([0.485, 0.456, 0.406])
    stddev_vec = np.array([0.229, 0.224, 0.225])
    norm_img_data = np.zeros(img_data.shape).astype('float32')

    for i in range(img_data.shape[0]):
        # for each pixel in each channel, divide the value by 255 to get value between [0, 1] and then normalize
        norm_img_data[i, :, :] = (img_data[i, :, :] / 255 - mean_vec[i]) / stddev_vec[i]
    return norm_img_data


def get_img(batch=1):
    img_url = "https://github.com/dmlc/mxnet.js/blob/main/data/cat.png?raw=true"
    img_path = download_testdata(img_url, "cat.png", module="data")

    img_path = "dog.png"

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    in_data = np.asarray(img[:, :])

    # hwc to chw
    in_data = in_data.transpose((2, 0, 1))
    shape = in_data.shape
    in_data = preprocess(in_data)
    in_data = np.broadcast_to(in_data.astype("float32"), shape=(batch, *shape))
    return in_data


class TraceWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, inp):
        logits = self.model(inp)
        return self.softmax(logits)


def trace_torch_model(model):
    # trace model to jit variant of representation this will require for conversion to TVM and CoreML
    example_input = torch.rand(1, 3, 224, 224)
    return torch.jit.trace(torch_model, example_input)

# Download class labels (from a separate file)
def load_class_labels():
    synset_url = "".join(
        [
            "https://gist.githubusercontent.com/zhreshold/",
            "4d0b62f3d01426887599d4f7ede23ee5/raw/",
            "596b27d23537e5a1b5751d2b0481ef172f58b539/",
            "imagenet1000_clsid_to_human.txt",
        ]
    )
    synset_name = "imagenet1000_clsid_to_human.txt"
    synset_path = download_testdata(synset_url, synset_name, module="data")
    with open(synset_path) as f:
        synset = eval(f.read())

    return synset


if __name__ == '__main__':
    # Load a pre-trained model
    torch_model = torchvision.models.squeezenet1_0(pretrained=True)
    # Set the model in evaluation mode
    torch_model = TraceWrapper(torch_model.eval())
    traced_model = trace_torch_model(torch_model)

    # get data
    class_labels = load_class_labels()
    data = get_img()

    # infer model with Torch
    with torch.no_grad():
        res = torch_model(torch.Tensor(data))
    res5 = res[0].argsort().numpy()[-5:][::-1]

    t = PrettyTable(['Pytorch Results', 'Label', 'Class ID', 'Probability'], float_format=".2")
    for i, idx in enumerate(res5):
        t.add_row([f"Top{i + 1}", class_labels[idx], idx, res[0][res5[i]].numpy() * 100])
    print(t)


    # compile and infer with TVM
    shape_list = [('data', [1, 3, 224, 224])]
    mod, params = relay.frontend.from_pytorch(traced_model, shape_list)

    #compile for iOS


    arch = "arm64"
    sdk = "iphoneos"
    target = f"llvm -model=iphone12mini -mtriple={arch}-apple-darwin -mattr=+neon"
    target_host = target

    log_file = os.path.join(os.getcwd(), "statistic/squeeznet_iphone_log.txt")

    with auto_scheduler.ApplyHistoryBest(log_file):
        with tvm.transform.PassContext(opt_level=3, config={"relay.backend.use_auto_scheduler": True}):
            lib = relay.build(mod, target=target, target_host=target_host, params=params)

    exported_model_file_name = os.path.join(os.getcwd(),"xOS.perf.benchmarking/iOS/compiled_model.dylib")

    builder = xcode.create_dylib
    lib.export_library(exported_model_file_name, fcompile=builder, arch=arch, sdk=sdk)

    #compile for macOS
    target = "llvm"
    target_host = target

    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target=target, target_host=target_host, params=params)

    exported_model_file_name = os.path.join(os.getcwd(),"xOS.perf.benchmarking/macOS/compiled_model.dylib")

    lib.export_library(exported_model_file_name)


    # inference through TVM on host macOS platform
    ctx = tvm.cpu(0)
    module = graph_runtime.graph_executor.GraphModule(lib["default"](ctx))

    data = get_img()

    module.set_input("data", data)
    module.run()

    res = module.get_output(0).asnumpy()[0]

    # cut last 5 values after argsort and reverse
    res5 = res.argsort()[-5:][::-1]

    t = PrettyTable(['TVM Results', 'Label', 'Class ID', 'Probability'], float_format=".2")
    for i, idx in enumerate(res5):
        t.add_row([f"Top{i + 1}", class_labels[idx], idx, res[idx] * 100])
    print(t)