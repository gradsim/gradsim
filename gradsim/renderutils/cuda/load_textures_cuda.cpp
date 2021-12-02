// Copyright (c) 2019-2021, NVIDIA CORPORATION. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <torch/torch.h>

// CUDA forward declarations

at::Tensor load_textures_cuda(
        at::Tensor image,
        at::Tensor faces,
        at::Tensor textures,
        at::Tensor is_update);

// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


at::Tensor load_textures(
        at::Tensor image,
        at::Tensor faces,
        at::Tensor textures,
        at::Tensor is_update) {

    CHECK_INPUT(image);
    CHECK_INPUT(faces);
    CHECK_INPUT(is_update);
    CHECK_INPUT(textures);

    return load_textures_cuda(image, faces, textures, is_update);
                                      
}

//PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
PYBIND11_MODULE(load_textures, m) {
    m.def("load_textures", &load_textures, "LOAD_TEXTURES (CUDA)");
}
