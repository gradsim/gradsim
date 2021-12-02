// This code contains NVIDIA Confidential Information and is disclosed to you
// under a form of NVIDIA software license agreement provided separately to you.
//
// Notice
// NVIDIA Corporation and its licensors retain all intellectual property and
// proprietary rights in and to this software and related documentation and
// any modifications thereto. Any use, reproduction, disclosure, or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA Corporation is strictly prohibited.
//
// ALL NVIDIA DESIGN SPECIFICATIONS, CODE ARE PROVIDED "AS IS.". NVIDIA MAKES
// NO WARRANTIES, EXPRESSED, IMPLIED, STATUTORY, OR OTHERWISE WITH RESPECT TO
// THE MATERIALS, AND EXPRESSLY DISCLAIMS ALL IMPLIED WARRANTIES OF NONINFRINGEMENT,
// MERCHANTABILITY, AND FITNESS FOR A PARTICULAR PURPOSE.
//
// Information and code furnished is believed to be accurate and reliable.
// However, NVIDIA Corporation assumes no responsibility for the consequences of use of such
// information or for any infringement of patents or other rights of third parties that may
// result from its use. No license is granted by implication or otherwise under any patent
// or patent rights of NVIDIA Corporation. Details are subject to change without notice.
// This code supersedes and replaces all information previously supplied.
// NVIDIA Corporation products are not authorized for use as critical
// components in life support devices or systems without express written approval of
// NVIDIA Corporation.
//
// Copyright (c) 2020-2021 NVIDIA Corporation. All rights reserved.

#pragma once

#ifdef CPU
    #define CUDA_CALLABLE
    #define __device__
    #define __host__
    #define __constant__
#elif defined(CUDA)
    #define CUDA_CALLABLE __device__ 
#endif

#define FP_CHECK 0

namespace df
{

#include <cmath>

#define kEps 0.0f


// basic ops for integer types
inline CUDA_CALLABLE int mul(int a, int b) { return a*b; }
inline CUDA_CALLABLE int div(int a, int b) { return a/b; }
inline CUDA_CALLABLE int add(int a, int b) { return a+b; }
inline CUDA_CALLABLE int sub(int a, int b) { return a-b; }
inline CUDA_CALLABLE int mod(int a, int b) { return a % b; }

inline CUDA_CALLABLE void adj_mul(int a, int b, int& adj_a, int& adj_b, int adj_ret) { }
inline CUDA_CALLABLE void adj_div(int a, int b, int& adj_a, int& adj_b, int adj_ret) { }
inline CUDA_CALLABLE void adj_add(int a, int b, int& adj_a, int& adj_b, int adj_ret) { }
inline CUDA_CALLABLE void adj_sub(int a, int b, int& adj_a, int& adj_b, int adj_ret) { }
inline CUDA_CALLABLE void adj_mod(int a, int b, int& adj_a, int& adj_b, int adj_ret) { }

// basic ops for float types
inline CUDA_CALLABLE float mul(float a, float b) { return a*b; }
inline CUDA_CALLABLE float div(float a, float b) { return a/b; }
inline CUDA_CALLABLE float add(float a, float b) { return a+b; }
inline CUDA_CALLABLE float sub(float a, float b) { return a-b; }
inline CUDA_CALLABLE float min(float a, float b) { return a<b?a:b; }
inline CUDA_CALLABLE float max(float a, float b) { return a>b?a:b; }
inline CUDA_CALLABLE float leaky_min(float a, float b, float r) { return min(a, b); }
inline CUDA_CALLABLE float leaky_max(float a, float b, float r) { return max(a, b); }
inline CUDA_CALLABLE float clamp(float x, float a, float b) { return min(max(a, x), b); }
inline CUDA_CALLABLE float step(float x) { return x < 0.0 ? 1.0 : 0.0; }
inline CUDA_CALLABLE float sign(float x) { return x < 0.0 ? -1.0 : 1.0; }
inline CUDA_CALLABLE float abs(float x) { return fabsf(x); }
inline CUDA_CALLABLE float nonzero(float x) { return x == 0.0 ? 0.0 : 1.0; }

inline CUDA_CALLABLE float acos(float x) { return std::acos(std::min(std::max(x, -1.0f), 1.0f)); }
inline CUDA_CALLABLE float sin(float x) { return std::sin(x); }
inline CUDA_CALLABLE float cos(float x) { return std::cos(x); }

inline CUDA_CALLABLE void adj_mul(float a, float b, float& adj_a, float& adj_b, float adj_ret) { adj_a += b*adj_ret; adj_b += a*adj_ret; }
inline CUDA_CALLABLE void adj_div(float a, float b, float& adj_a, float& adj_b, float adj_ret) { adj_a += adj_ret/b; adj_b -= adj_ret*a/(b*b); }
inline CUDA_CALLABLE void adj_add(float a, float b, float& adj_a, float& adj_b, float adj_ret) { adj_a += adj_ret; adj_b += adj_ret; }
inline CUDA_CALLABLE void adj_sub(float a, float b, float& adj_a, float& adj_b, float adj_ret) { adj_a += adj_ret; adj_b -= adj_ret; }

// inline CUDA_CALLABLE bool lt(float a, float b) { return a < b; }
// inline CUDA_CALLABLE bool gt(float a, float b) { return a > b; }
// inline CUDA_CALLABLE bool lte(float a, float b) { return a <= b; }
// inline CUDA_CALLABLE bool gte(float a, float b) { return a >= b; }
// inline CUDA_CALLABLE bool eq(float a, float b) { return a == b; }
// inline CUDA_CALLABLE bool neq(float a, float b) { return a != b; }

// inline CUDA_CALLABLE bool adj_lt(float a, float b, float & adj_a, float & adj_b, bool & adj_ret) { }
// inline CUDA_CALLABLE bool adj_gt(float a, float b, float & adj_a, float & adj_b, bool & adj_ret) {  }
// inline CUDA_CALLABLE bool adj_lte(float a, float b, float & adj_a, float & adj_b, bool & adj_ret) {  }
// inline CUDA_CALLABLE bool adj_gte(float a, float b, float & adj_a, float & adj_b, bool & adj_ret) {  }
// inline CUDA_CALLABLE bool adj_eq(float a, float b, float & adj_a, float & adj_b, bool & adj_ret) {  }
// inline CUDA_CALLABLE bool adj_neq(float a, float b, float & adj_a, float & adj_b, bool & adj_ret) {  }

inline CUDA_CALLABLE void adj_min(float a, float b, float& adj_a, float& adj_b, float adj_ret)
{
    if (a < b)
        adj_a += adj_ret;
    else
        adj_b += adj_ret;
}

inline CUDA_CALLABLE void adj_max(float a, float b, float& adj_a, float& adj_b, float adj_ret)
{
    if (a > b)
        adj_a += adj_ret;
    else
        adj_b += adj_ret;
}

inline CUDA_CALLABLE void adj_leaky_min(float a, float b, float r, float& adj_a, float& adj_b, float& adj_r, float adj_ret)
{
    if (a < b)
        adj_a += adj_ret;
    else
    {
        adj_a += r*adj_ret;
        adj_b += adj_ret;
    }
}

inline CUDA_CALLABLE void adj_leaky_max(float a, float b, float r, float& adj_a, float& adj_b, float& adj_r, float adj_ret)
{
    if (a > b)
        adj_a += adj_ret;
    else
    {
        adj_a += r*adj_ret;
        adj_b += adj_ret;
    }
}

inline CUDA_CALLABLE void adj_clamp(float x, float a, float b, float& adj_x, float& adj_a, float& adj_b, float adj_ret)
{
    if (x < a)
        adj_a += adj_ret;
    else if (x > b)
        adj_b += adj_ret;
    else
        adj_x += adj_ret;
}

inline CUDA_CALLABLE void adj_step(float x, float& adj_x, float adj_ret)
{
    // nop
}

inline CUDA_CALLABLE void adj_nonzero(float x, float& adj_x, float adj_ret)
{
    // nop
}

inline CUDA_CALLABLE void adj_sign(float x, float& adj_x, float adj_ret)
{
    // nop
}

inline CUDA_CALLABLE void adj_abs(float x, float& adj_x, float adj_ret)
{
    if (x < 0.0)
        adj_x += adj_ret;
    else
        adj_x -= adj_ret;                
}

inline CUDA_CALLABLE void adj_acos(float x, float& adj_x, float adj_ret)
{
    float d = sqrt(1.0-x*x);
    if (d > 0.0)
        adj_x -= (1.0/d)*adj_ret;
}

inline CUDA_CALLABLE void adj_sin(float x, float& adj_x, float adj_ret)
{
    adj_x += std::cos(x)*adj_ret;
}

inline CUDA_CALLABLE void adj_cos(float x, float& adj_x, float adj_ret)
{
    adj_x -= std::sin(x)*adj_ret;
}

template <typename T>
CUDA_CALLABLE inline T select(bool cond, const T& a, const T& b) { return cond?b:a; }

template <typename T>
CUDA_CALLABLE inline void adj_select(bool cond, const T& a, const T& b, bool& adj_cond, T& adj_a, T& adj_b, const T& adj_ret)
{
    if (cond)
        adj_b += adj_ret;
    else
        adj_a += adj_ret;
}



// some helpful operator overloads (just for C++ use, these are not adjointed)

template <typename T>
CUDA_CALLABLE T& operator += (T& a, const T& b) { a = add(a, b); return a; }

template <typename T>
CUDA_CALLABLE T& operator -= (T& a, const T& b) { a = sub(a, b); return a; }

template <typename T>
CUDA_CALLABLE T operator*(const T& a, float s) { return mul(a, s); }

template <typename T>
CUDA_CALLABLE T operator/(const T& a, float s) { return div(a, s); }

template <typename T>
CUDA_CALLABLE T operator+(const T& a, const T& b) { return add(a, b); }

template <typename T>
CUDA_CALLABLE T operator-(const T& a, const T& b) { return sub(a, b); }


#include "vec2.h"
#include "vec3.h"
#include "mat22.h"
#include "mat33.h"
#include "quat.h"
#include "spatial.h"

//--------------

// for single thread CPU only
static int s_threadIdx;

inline CUDA_CALLABLE int tid()
{
#ifdef CPU
    return s_threadIdx;
#elif defined(CUDA)
    return blockDim.x * blockIdx.x + threadIdx.x;
#endif
}

template<typename T>
inline CUDA_CALLABLE T load(T* buf, int index)
{
    assert(buf);
    return buf[index];
}

template<typename T>
inline CUDA_CALLABLE void store(T* buf, int index, T value)
{
    // allow NULL buffers for case where gradients are not required
    if (buf)
    {
        buf[index] = value;
    }
}

#ifdef CUDA

template<typename T>
inline __device__ void atomic_add(T* buf, T value)
{
    atomicAdd(buf, value);
}
#endif

template<typename T>
inline __device__ void atomic_add(T* buf, int index, T value)
{
    if (buf)
    {
        // CPU mode is sequential so just add
#ifdef CPU
        buf[index] += value;
#elif defined(CUDA)
        atomic_add(buf + index, value);
#endif
    }
}

template<typename T>
inline __device__ void atomic_sub(T* buf, int index, T value)
{
    if (buf)
    {
        // CPU mode is sequential so just add
#ifdef CPU
        buf[index] -= value;
#elif defined(CUDA)
        atomic_add(buf + index, -value);
#endif
    }
}

template <typename T>
inline CUDA_CALLABLE void adj_load(T* buf, int index, T* adj_buf, int& adj_index, const T& adj_output)
{
    // allow NULL buffers for case where gradients are not required
    if (adj_buf) {
#ifdef CPU
        adj_buf[index] += adj_output;  // does not need to be atomic if single-threaded
#elif defined(CUDA)
        atomic_add(adj_buf, index, adj_output);
#endif
        
    }
}

template <typename T>
inline CUDA_CALLABLE void adj_store(T* buf, int index, T value, T* adj_buf, int& adj_index, T& adj_value)
{   
    adj_value += adj_buf[index]; // doesn't need to be atomic because it's used to load from a buffer onto the stack
}

template<typename T>
inline CUDA_CALLABLE void adj_atomic_add(T* buf, int index, T value, T* adj_buf, int& adj_index, T& adj_value)
{
    if (adj_buf) {  // cannot be atomic because used locally
        adj_value += adj_buf[index];
    }
}

template<typename T>
inline CUDA_CALLABLE void adj_atomic_sub(T* buf, int index, T value, T* adj_buf, int& adj_index, T& adj_value)
{
    if (adj_buf) { // cannot be atomic because used locally
        adj_value -= adj_buf[index];
    }
}

//-------------------------
// Texture methods

inline CUDA_CALLABLE float sdf_sample(float3 x)
{
    return 0.0;
}

inline CUDA_CALLABLE float3 sdf_grad(float3 x)
{
    return float3();
}

inline CUDA_CALLABLE void adj_sdf_sample(float3 x, float3& adj_x, float adj_ret)
{

}

inline CUDA_CALLABLE void adj_sdf_grad(float3 x, float3& adj_x, float3& adj_ret)
{

}

inline CUDA_CALLABLE void print(int i)
{
    printf("%d\n", i);
}

inline CUDA_CALLABLE void print(float i)
{
    printf("%f\n", i);
}

inline CUDA_CALLABLE void print(float3 i)
{
    printf("%f %f %f\n", i.x, i.y, i.z);
}

inline CUDA_CALLABLE void print(quat i)
{
    printf("%f %f %f %f\n", i.x, i.y, i.z, i.w);
}

inline CUDA_CALLABLE void print(mat22 m)
{
    printf("%f %f\n%f %f\n", m.data[0][0], m.data[0][1], 
                             m.data[1][0], m.data[1][1]);
}

inline CUDA_CALLABLE void print(mat33 m)
{
    printf("%f %f %f\n%f %f %f\n%f %f %f\n", m.data[0][0], m.data[0][1], m.data[0][2], 
                                             m.data[1][0], m.data[1][1], m.data[1][2], 
                                             m.data[2][0], m.data[2][1], m.data[2][2]);
}

inline CUDA_CALLABLE void print(spatial_transform t)
{
    printf("(%f %f %f) (%f %f %f %f)\n", t.p.x, t.p.y, t.p.z, t.q.x, t.q.y, t.q.z, t.q.w);
}

inline CUDA_CALLABLE void adj_print(int i, int& adj_i) { printf("%d adj: %d\n", i, adj_i); }
inline CUDA_CALLABLE void adj_print(float i, float& adj_i) { printf("%f adj: %f\n", i, adj_i); }
inline CUDA_CALLABLE void adj_print(float3 i, float3& adj_i) { printf("%f %f %f adj: %f %f %f \n", i.x, i.y, i.z, adj_i.x, adj_i.y, adj_i.z); }
inline CUDA_CALLABLE void adj_print(quat i, quat& adj_i) { }
inline CUDA_CALLABLE void adj_print(mat22 m, mat22& adj_m) { }
inline CUDA_CALLABLE void adj_print(mat33 m, mat33& adj_m) { }
inline CUDA_CALLABLE void adj_print(spatial_transform t, spatial_transform& adj_t) {}

} // namespace df