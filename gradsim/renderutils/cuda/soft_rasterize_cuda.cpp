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

#include <iostream>

// CUDA forward declarations


std::vector<at::Tensor> forward_soft_rasterize_cuda(
        at::Tensor faces,
        at::Tensor textures,
        at::Tensor faces_info,
        at::Tensor aggrs_info,
        at::Tensor soft_colors,
        int image_size,
        float near,
        float far,
        float eps,
        float sigma_val,
        int func_id_dist,
        float dist_eps,
        float gamma_val,
        int func_id_rgb,
        int func_id_alpha,
        int texture_sample_type,
        bool double_side);


std::vector<at::Tensor> backward_soft_rasterize_cuda(
        at::Tensor faces,
        at::Tensor textures,
        at::Tensor soft_colors,        
        at::Tensor faces_info,
        at::Tensor aggrs_info,
        at::Tensor grad_faces,
        at::Tensor grad_textures,
        at::Tensor grad_soft_colors,
        int image_size,
        float near,
        float far,
        float eps,
        float sigma_val,
        int func_id_dist,
        float dist_eps,
        float gamma_val,
        int func_id_rgb,
        int func_id_alpha,
        int texture_sample_type,
        bool double_side);

// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


std::vector<at::Tensor> forward_soft_rasterize(
        at::Tensor faces,
        at::Tensor textures,
        at::Tensor faces_info,
        at::Tensor aggrs_info,
        at::Tensor soft_colors,
        int image_size,
        float near,
        float far,
        float eps,
        float sigma_val,
        int func_id_dist,
        float dist_eps,
        float gamma_val,
        int func_id_rgb,
        int func_id_alpha,
        int texture_sample_type,
        bool double_side) {

    CHECK_INPUT(faces);
    CHECK_INPUT(textures);
    CHECK_INPUT(faces_info);
    CHECK_INPUT(aggrs_info);
    CHECK_INPUT(soft_colors);

    return forward_soft_rasterize_cuda(faces, textures, 
                                       faces_info, aggrs_info, 
                                       soft_colors, 
                                       image_size, near, far, eps, 
                                       sigma_val, func_id_dist, dist_eps,
                                       gamma_val, func_id_rgb, func_id_alpha, 
                                       texture_sample_type, double_side);
}


std::vector<at::Tensor> backward_soft_rasterize(
        at::Tensor faces,
        at::Tensor textures,
        at::Tensor soft_colors,
        at::Tensor faces_info,
        at::Tensor aggrs_info,
        at::Tensor grad_faces,
        at::Tensor grad_textures,
        at::Tensor grad_soft_colors,
        int image_size,
        float near,
        float far,
        float eps,
        float sigma_val,
        int func_id_dist,
        float dist_eps,
        float gamma_val,
        int func_id_rgb,
        int func_id_alpha,
        int texture_sample_type,
        bool double_side) {

    CHECK_INPUT(faces);
    CHECK_INPUT(textures);
    CHECK_INPUT(soft_colors);
    CHECK_INPUT(faces_info);
    CHECK_INPUT(aggrs_info);
    CHECK_INPUT(grad_faces);
    CHECK_INPUT(grad_textures);
    CHECK_INPUT(grad_soft_colors);

    return backward_soft_rasterize_cuda(faces, textures, soft_colors, 
                                        faces_info, aggrs_info, 
                                        grad_faces, grad_textures, grad_soft_colors, 
                                        image_size, near, far, eps, 
                                        sigma_val, func_id_dist, dist_eps,
                                        gamma_val, func_id_rgb, func_id_alpha,
                                        texture_sample_type, double_side);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward_soft_rasterize, "SOFT_RASTERIZE_FORWARD (CUDA)");
    m.def("backward", &backward_soft_rasterize, "SOFT_RASTERIZE_BACKWARD (CUDA)");
}
