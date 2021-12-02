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

//----------------------------------------------------------
// mat33

struct mat33
{
    inline CUDA_CALLABLE mat33(float3 c0, float3 c1, float3 c2)
    {
        data[0][0] = c0.x;
        data[1][0] = c0.y;
        data[2][0] = c0.z;

        data[0][1] = c1.x;
        data[1][1] = c1.y;
        data[2][1] = c1.z;

        data[0][2] = c2.x;
        data[1][2] = c2.y;
        data[2][2] = c2.z;
    }

    inline CUDA_CALLABLE mat33(float m00=0.0f, float m01=0.0f, float m02=0.0f,
                 float m10=0.0f, float m11=0.0f, float m12=0.0f,
                 float m20=0.0f, float m21=0.0f, float m22=0.0f) 
    {
        data[0][0] = m00;
        data[1][0] = m10;
        data[2][0] = m20;

        data[0][1] = m01;
        data[1][1] = m11;
        data[2][1] = m21;

        data[0][2] = m02;
        data[1][2] = m12;
        data[2][2] = m22;
    }

    CUDA_CALLABLE float3 get_row(int index) const
    {
        return (float3&)data[index]; 
    }

    CUDA_CALLABLE void set_row(int index, const float3& v)
    {
        (float3&)data[index] = v;
    }

    CUDA_CALLABLE float3 get_col(int index) const
    {
        return float3(data[0][index], data[1][index], data[2][index]);
    }

    CUDA_CALLABLE void set_col(int index, const float3& v)
    {
        data[0][index] = v.x;
        data[1][index] = v.y;
        data[2][index] = v.z;
    }

    // row major storage assumed to be compatible with PyTorch
    float data[3][3];
};

#ifdef CUDA
inline __device__ void atomic_add(mat33 * addr, mat33 value) {
    atomicAdd(&((addr -> data)[0][0]), value.data[0][0]);
    atomicAdd(&((addr -> data)[1][0]), value.data[1][0]);
    atomicAdd(&((addr -> data)[2][0]), value.data[2][0]);
    atomicAdd(&((addr -> data)[0][1]), value.data[0][1]);
    atomicAdd(&((addr -> data)[1][1]), value.data[1][1]);
    atomicAdd(&((addr -> data)[2][1]), value.data[2][1]);
    atomicAdd(&((addr -> data)[0][2]), value.data[0][2]);
    atomicAdd(&((addr -> data)[1][2]), value.data[1][2]);
    atomicAdd(&((addr -> data)[2][2]), value.data[2][2]);
}
#endif

inline CUDA_CALLABLE void adj_mat33(float3 c0, float3 c1, float3 c2,
                      float3& a0, float3& a1, float3& a2,
                      const mat33& adj_ret)
{
    // column constructor
    a0 += adj_ret.get_col(0);
    a1 += adj_ret.get_col(1);
    a2 += adj_ret.get_col(2);

}

inline CUDA_CALLABLE void adj_mat33(float m00, float m01, float m02,
                      float m10, float m11, float m12,
                      float m20, float m21, float m22,
                      float& a00, float& a01, float& a02,
                      float& a10, float& a11, float& a12,
                      float& a20, float& a21, float& a22,
                      const mat33& adj_ret)
{
    printf("todo\n");
}

inline CUDA_CALLABLE float index(const mat33& m, int row, int col)
{
    return m.data[row][col];
}

inline CUDA_CALLABLE mat33 add(const mat33& a, const mat33& b)
{
    mat33 t;
    for (int i=0; i < 3; ++i)
    {
        for (int j=0; j < 3; ++j)
        {
            t.data[i][j] = a.data[i][j] + b.data[i][j];
        }
    }

    return t;
}

inline CUDA_CALLABLE mat33 mul(const mat33& a, float b)
{
    mat33 t;
    for (int i=0; i < 3; ++i)
    {
        for (int j=0; j < 3; ++j)
        {
            t.data[i][j] = a.data[i][j]*b;
        }
    }

    return t;   
}


inline CUDA_CALLABLE float3 mul(const mat33& a, const float3& b)
{
    float3 r = a.get_col(0)*b.x +
               a.get_col(1)*b.y +
               a.get_col(2)*b.z;
    
    return r;
}

inline CUDA_CALLABLE mat33 mul(const mat33& a, const mat33& b)
{
    mat33 t;
    for (int i=0; i < 3; ++i)
    {
        for (int j=0; j < 3; ++j)
        {
            for (int k=0; k < 3; ++k)
            {
                t.data[i][j] += a.data[i][k]*b.data[k][j];
            }
        }
    }

    return t;
}

inline CUDA_CALLABLE mat33 transpose(const mat33& a)
{
    mat33 t;
    for (int i=0; i < 3; ++i)
    {
        for (int j=0; j < 3; ++j)
        {
            t.data[i][j] = a.data[j][i];
        }
    }

    return t;
}


inline CUDA_CALLABLE float determinant(const mat33& m)
{
    return dot(float3(m.data[0]), cross(float3(m.data[1]), float3(m.data[2])));
}

inline CUDA_CALLABLE mat33 outer(const float3& a, const float3& b)
{
    return mat33(a*b.x, a*b.y, a*b.z);    
}

inline void CUDA_CALLABLE adj_index(const mat33& m, int row, int col, mat33& adj_m, int& adj_row, int& adj_col, float adj_ret)
{
    adj_m.data[row][col] += adj_ret;
}

inline CUDA_CALLABLE void adj_add(const mat33& a, const mat33& b, mat33& adj_a, mat33& adj_b, const mat33& adj_ret)
{
    for (int i=0; i < 3; ++i)
    {
        for (int j=0; j < 3; ++j)
        {
            adj_a.data[i][j] = adj_ret.data[i][j];
            adj_b.data[i][j] = adj_ret.data[i][j];
        }
    }
}

inline CUDA_CALLABLE void adj_mul(const mat33& a, float b, mat33& adj_a, float& adj_b, const mat33& adj_ret)
{
    for (int i=0; i < 3; ++i)
    {
        for (int j=0; j < 3; ++j)
        {
            adj_a.data[i][j] += b*adj_ret.data[i][j];
            adj_b += a.data[i][j]*adj_ret.data[i][j];
        }
    }
}

inline CUDA_CALLABLE void adj_mul(const mat33& a, const float3& b, mat33& adj_a, float3& adj_b, const float3& adj_ret)
{
    adj_a += outer(adj_ret, b);
    adj_b += mul(transpose(a), adj_ret);
}

inline CUDA_CALLABLE void adj_mul(const mat33& a, const mat33& b, mat33& adj_a, mat33& adj_b, const mat33& adj_ret)
{
    adj_a += mul(adj_ret, transpose(b));
    adj_b += mul(transpose(a), adj_ret);
}

inline CUDA_CALLABLE void adj_transpose(const mat33& a, mat33& adj_a, const mat33& adj_ret)
{
    adj_a += transpose(adj_ret);
}

inline CUDA_CALLABLE void adj_determinant(const mat33& m, mat33& adj_m, float adj_ret)
{
    (float3&)adj_m.data[0] += cross(m.get_row(1), m.get_row(2))*adj_ret;
    (float3&)adj_m.data[1] += cross(m.get_row(2), m.get_row(0))*adj_ret;
    (float3&)adj_m.data[2] += cross(m.get_row(0), m.get_row(1))*adj_ret;
}
