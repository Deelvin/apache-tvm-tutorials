/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

import SwiftUI
import Foundation
import CoreML

import SwiftUI
import UniformTypeIdentifiers

#if !os(iOS)
import Cocoa
#endif


class PerformanceDataModel : ObservableObject {
    @Published var image_filename: String = "Not selected"
    @Published var filename: String = "Not selected"
    @Published var model_filename: String = "Squeezenet1.1"
    @Published var probability: String = "--.--"
    @Published var perf_data: String = "---.--"
    @Published var top1: String = "Top1 class"
    @Published var fileUrl: URL = URL(fileURLWithPath: "")
    
    
    func doInfer() {

        #if os(iOS)
            self.fileUrl.startAccessingSecurityScopedResource()
            let imageData = (try? Data(contentsOf: self.fileUrl))!
            let image = UIImage(data: imageData)!
            let pixelBuffer = image.pixelBuffer(width: 224, height: 224)
            self.fileUrl.stopAccessingSecurityScopedResource()
        #else
            let image = NSImage(byReferencing: self.fileUrl)
            let resized_image = image.resize(withSize: NSSize(width: 224, height: 224))
            let pixelBuffer = resized_image?.pixelBuffer()!
        #endif
        
            CVPixelBufferLockBaseAddress(pixelBuffer!, CVPixelBufferLockFlags(rawValue: 0))
            let baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer!)
            let width = CVPixelBufferGetWidth(pixelBuffer!)
            let height = CVPixelBufferGetHeight(pixelBuffer!)
     
            let handle = createTVMInferWrapper()
            setTVMInputParams(handle, 1, 3, Int32(height), Int32(width))
            
            var result = TVMClassificationResult()
            doTVMinfer(handle, UnsafeMutablePointer<UInt32>(OpaquePointer(baseAddress)), &result)
            
            CVPixelBufferUnlockBaseAddress(pixelBuffer!, CVPixelBufferLockFlags(rawValue: 0))

            self.probability = String(format: "%3.2f%", result.probability*100)
            self.top1 = imagenet_labels[Int(result.class_id)]
            
            self.perf_data = String(format: "%3.2f%", result.performance)
        
            removeTVMInferWrapper(handle)
    }
}


struct ImageInferenceView: View {
    @ObservedObject var viewModel : PerformanceDataModel = PerformanceDataModel()
    @State var openFile = false
    @State private var fileSelected = false


    func getImage() -> Image {
        
        if viewModel.image_filename == "Not selected" {
            return Image("background")
        } else {
        
        #if os(macOS)
            let image = NSImage(byReferencing: viewModel.fileUrl)
            let resized_image = image.resize(withSize: NSSize(width: 224, height: 224))
            
            return Image(nsImage: resized_image!)
        #else
            
            if viewModel.fileUrl.startAccessingSecurityScopedResource() == true {
                let imageData = (try? Data(contentsOf: viewModel.fileUrl))!
                let image = UIImage(data: imageData)!
                let resized_image = image.resized(to: CGSize(width: 224, height: 224))
                
                return Image(uiImage: resized_image)
            }
            return Image("background")

        #endif
        }
    }
    

    var body: some View {
            
            VStack {
                Text("Model")
                    .font(.title3)
            
                HStack {
                    Text(viewModel.model_filename)
                    Spacer()
                    Text("Pytorch ML Zoo")
                }
                .font(.subheadline)
                .foregroundColor(.secondary)

                Divider()
                
                Text("Performance")
                  .font(.title3)
                HStack {
                    Text("\(viewModel.perf_data)")
                    Spacer()
                    Text("fps")
                }
                .font(.subheadline)
                .foregroundColor(.secondary)
              
                Divider()

                Text("Top1")
                  .font(.title3)
                HStack {
                    Text("\(viewModel.top1)")
                    Spacer()
                    Text("\(viewModel.probability)%")
                }
                .font(.subheadline)
                .foregroundColor(.secondary)

                
                VStack {
                    Divider()
                    HStack() {
                        Button(action: {
                            openFile.toggle()
                        }) {
                            Text("Select file")
                        }
                        Spacer()
                        Text(viewModel.filename)
                    }
                    .fileImporter(isPresented: $openFile, allowedContentTypes: [.png, .bmp, .jpeg]) { (result) in
                        do {
                            let fileUrl = try result.get()
                            viewModel.fileUrl = fileUrl
                            viewModel.image_filename = fileUrl.absoluteString
                            viewModel.filename = fileUrl.lastPathComponent
                            
                            fileSelected = true
                         
                        } catch {
                            print("Error reading image")
                            print(error.localizedDescription)
                        }
                    }
                    getImage().shadow(radius: 8)
                    Divider()
                    Button("Run inference", action: doInfer)
                        .disabled(fileSelected == false)
                }
        }

    }


    func doInfer() {
        viewModel.doInfer()
    }
        
}


struct ImageInferenceView_Previews: PreviewProvider {
    static var previews: some View {
        ImageInferenceView()
    }
}
