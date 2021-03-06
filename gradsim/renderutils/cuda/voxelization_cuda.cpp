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

#include <vector>

// CUDA forward declarations

std::vector<at::Tensor> voxelize_sub1_cuda(
        at::Tensor faces,
        at::Tensor voxels);


std::vector<at::Tensor> voxelize_sub2_cuda(
        at::Tensor faces,
        at::Tensor voxels);


std::vector<at::Tensor> voxelize_sub3_cuda(
        at::Tensor faces,
        at::Tensor voxels,
        at::Tensor visible);


std::vector<at::Tensor> voxelize_sub4_cuda(
        at::Tensor faces,
        at::Tensor voxels,
        at::Tensor visible);



// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


std::vector<at::Tensor> voxelize_sub1(
        at::Tensor faces,
        at::Tensor voxels) {
    CHECK_INPUT(faces);
    CHECK_INPUT(voxels);

    return voxelize_sub1_cuda(faces, voxels);
}

std::vector<at::Tensor> voxelize_sub2(
        at::Tensor faces,
        at::Tensor voxels) {
    CHECK_INPUT(faces);
    CHECK_INPUT(voxels);

    return voxelize_sub2_cuda(faces, voxels);
}

std::vector<at::Tensor> voxelize_sub3(
        at::Tensor faces,
        at::Tensor voxels,
        at::Tensor visible) {
    CHECK_INPUT(faces);
    CHECK_INPUT(voxels);
    CHECK_INPUT(visible);

    return voxelize_sub3_cuda(faces, voxels, visible);
}

std::vector<at::Tensor> voxelize_sub4(
        at::Tensor faces,
        at::Tensor voxels,
        at::Tensor visible) {
    CHECK_INPUT(faces);
    CHECK_INPUT(voxels);
    CHECK_INPUT(visible);

    return voxelize_sub4_cuda(faces, voxels, visible);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("voxelize_sub1", &voxelize_sub1, "VOXELIZE_SUB1 (CUDA)");
    m.def("voxelize_sub2", &voxelize_sub2, "VOXELIZE_SUB2 (CUDA)");
    m.def("voxelize_sub3", &voxelize_sub3, "VOXELIZE_SUB3 (CUDA)");
    m.def("voxelize_sub4", &voxelize_sub4, "VOXELIZE_SUB4 (CUDA)");
}
