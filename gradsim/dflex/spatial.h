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

//---------------------------------------------------------------------------------
// Represents a twist in se(3)

struct spatial_vector
{
    float3 w;
    float3 v;

    CUDA_CALLABLE inline spatial_vector(float a, float b, float c, float d, float e, float f) : w(a, b, c), v(d, e, f) {}
    CUDA_CALLABLE inline spatial_vector(float3 w=float3(), float3 v=float3()) : w(w), v(v) {}
    CUDA_CALLABLE inline spatial_vector(float a) : w(a, a, a), v(a, a, a) {}
};


CUDA_CALLABLE inline spatial_vector add(const spatial_vector& a, const spatial_vector& b)
{
    return { a.w + b.w, a.v + b.v };
}

CUDA_CALLABLE inline spatial_vector sub(const spatial_vector& a, const spatial_vector& b)
{
    return { a.w - b.w, a.v - b.v };
}

CUDA_CALLABLE inline spatial_vector mul(const spatial_vector& a, float s)
{
    return { a.w*s, a.v*s };
}

CUDA_CALLABLE inline float spatial_dot(const spatial_vector& a, const spatial_vector& b)
{
    return dot(a.w, b.w) + dot(a.v, b.v);
}

CUDA_CALLABLE inline spatial_vector spatial_cross(const spatial_vector& a,  const spatial_vector& b)
{
    float3 w = cross(a.w, b.w);
    float3 v = cross(a.v, b.w) + cross(a.w, b.v);
    
    return spatial_vector(w, v);
}

CUDA_CALLABLE inline spatial_vector spatial_cross_dual(const spatial_vector& a,  const spatial_vector& b)
{
    float3 w = cross(a.w, b.w) + cross(a.v, b.v);
    float3 v = cross(a.w, b.v);

    return spatial_vector(w, v);
}


// adjoint methods
CUDA_CALLABLE inline void adj_spatial_vector(
    float a, float b, float c, 
    float d, float e, float f, 
    float& adj_a, float& adj_b, float& adj_c,
    float& adj_d, float& adj_e,float& adj_f, 
    const spatial_vector& adj_ret)
{
    adj_a += adj_ret.w.x;
    adj_b += adj_ret.w.y;
    adj_c += adj_ret.w.z;
    
    adj_d += adj_ret.v.x;
    adj_e += adj_ret.v.y;
    adj_f += adj_ret.v.z;
}

CUDA_CALLABLE inline void adj_spatial_vector(const float3& w, const float3& v, float3& adj_w, float3& adj_v, const spatial_vector& adj_ret)
{
    adj_w += adj_ret.w;
    adj_v += adj_ret.v;
}

CUDA_CALLABLE inline void adj_add(const spatial_vector& a, const spatial_vector& b, spatial_vector& adj_a, spatial_vector& adj_b, const spatial_vector& adj_ret)
{
    adj_add(a.w, b.w, adj_a.w, adj_b.w, adj_ret.w);
    adj_add(a.v, b.v, adj_a.v, adj_b.v, adj_ret.v);
}

CUDA_CALLABLE inline void adj_sub(const spatial_vector& a, const spatial_vector& b, spatial_vector& adj_a, spatial_vector& adj_b, const spatial_vector& adj_ret)
{
    adj_sub(a.w, b.w, adj_a.w, adj_b.w, adj_ret.w);
    adj_sub(a.v, b.v, adj_a.v, adj_b.v, adj_ret.v);
}

CUDA_CALLABLE inline void adj_mul(const spatial_vector& a, float s, spatial_vector& adj_a, float& adj_s, const spatial_vector& adj_ret)
{
    adj_mul(a.w, s, adj_a.w, adj_s, adj_ret.w);
    adj_mul(a.v, s, adj_a.v, adj_s, adj_ret.v);
}

CUDA_CALLABLE inline void adj_spatial_dot(const spatial_vector& a, const spatial_vector& b, spatial_vector& adj_a, spatial_vector& adj_b, const float& adj_ret)
{
    adj_dot(a.w, b.w, adj_a.w, adj_b.w, adj_ret);
    adj_dot(a.v, b.v, adj_a.v, adj_b.v, adj_ret);
}

CUDA_CALLABLE inline void adj_spatial_cross(const spatial_vector& a,  const spatial_vector& b, spatial_vector& adj_a,  spatial_vector& adj_b, const spatial_vector& adj_ret)
{
    adj_cross(a.w, b.w, adj_a.w, adj_b.w, adj_ret.w);
    
    adj_cross(a.v, b.w, adj_a.v, adj_b.w, adj_ret.v);
    adj_cross(a.w, b.v, adj_a.w, adj_b.v, adj_ret.v);
}

CUDA_CALLABLE inline void spatial_cross_dual(const spatial_vector& a,  const spatial_vector& b, spatial_vector& adj_a, spatial_vector& adj_b, const spatial_vector& adj_ret)
{
    adj_cross(a.w, b.w, adj_a.w, adj_b.w, adj_ret.w);
    adj_cross(a.v, b.v, adj_a.v, adj_b.v, adj_ret.w);

    adj_cross(a.w, b.v, adj_a.w, adj_b.v, adj_ret.v);
}


#ifdef CUDA
inline __device__ void atomic_add(spatial_vector* addr, const spatial_vector& value) {
    
    atomic_add(&addr->w, value.w);
    atomic_add(&addr->v, value.v);
}
#endif

//---------------------------------------------------------------------------------
// Represents a rigid body transformation

struct spatial_transform
{
    float3 p;
    quat q;

    CUDA_CALLABLE inline spatial_transform(float3 p=float3(), quat q=quat()) : p(p), q(q) {}
    CUDA_CALLABLE inline spatial_transform(float)  {}  // helps uniform initialization
};

CUDA_CALLABLE inline spatial_transform spatial_transform_identity()
{
    return spatial_transform(float3(), quat_identity());
}

CUDA_CALLABLE inline spatial_transform spatial_transform_multiply(const spatial_transform& a, const spatial_transform& b)
{
    return { rotate(a.q, b.p) + a.p, mul(a.q, b.q) };
}

CUDA_CALLABLE inline spatial_vector spatial_transform_twist(const spatial_transform& a, const spatial_vector& s)
{
    printf("todo");
    return spatial_vector();
}

CUDA_CALLABLE inline spatial_transform add(const spatial_transform& a, const spatial_transform& b)
{
    return { a.p + b.p, a.q + b.q };
}

CUDA_CALLABLE inline spatial_transform sub(const spatial_transform& a, const spatial_transform& b)
{
    return { a.p - b.p, a.q - b.q };
}

CUDA_CALLABLE inline spatial_transform mul(const spatial_transform& a, float s)
{
    return { a.p*s, a.q*s };
}


// adjoint methods
CUDA_CALLABLE inline void adj_add(const spatial_transform& a, const spatial_transform& b, spatial_transform& adj_a, spatial_transform& adj_b, const spatial_transform& adj_ret)
{
    adj_add(a.p, b.p, adj_a.p, adj_b.p, adj_ret.p);
    adj_add(a.q, b.q, adj_a.q, adj_b.q, adj_ret.q);
}

CUDA_CALLABLE inline void adj_sub(const spatial_transform& a, const spatial_transform& b, spatial_transform& adj_a, spatial_transform& adj_b, const spatial_transform& adj_ret)
{
    adj_sub(a.p, b.p, adj_a.p, adj_b.p, adj_ret.p);
    adj_sub(a.q, b.q, adj_a.q, adj_b.q, adj_ret.q);
}

CUDA_CALLABLE inline void adj_mul(const spatial_transform& a, float s, spatial_transform& adj_a, float& adj_s, const spatial_transform& adj_ret)
{
    adj_mul(a.p, s, adj_a.p, adj_s, adj_ret.p);
    adj_mul(a.q, s, adj_a.q, adj_s, adj_ret.q);
}

#ifdef CUDA
inline __device__ void atomic_add(spatial_transform* addr, const spatial_transform& value) {
    
    atomic_add(&addr->p, value.p);
    atomic_add(&addr->q, value.q);
}
#endif

CUDA_CALLABLE inline void adj_spatial_transform(const float3& p, const quat& q, float3& adj_p, quat& adj_q, const spatial_transform& adj_ret)
{
    adj_p += adj_ret.p;
    adj_q += adj_ret.q;
}

CUDA_CALLABLE inline void adj_spatial_transform_identity(const spatial_transform& adj_ret)
{
    printf("impl\n");
}

CUDA_CALLABLE inline void adj_spatial_transform_multiply(const spatial_transform& a, const spatial_transform& b, spatial_transform& adj_a, spatial_transform& adj_b, const spatial_transform& adj_ret)
{
    // translational part
    adj_rotate(a.q, b.p, adj_a.q, adj_b.p, adj_ret.p);
    adj_a.p += adj_ret.p;

    // rotational part
    adj_mul(a.q, b.q, adj_a.q, adj_b.q, adj_ret.q);
}

CUDA_CALLABLE inline void adj_spatial_transform_twist(const spatial_transform& a, const spatial_vector& s, spatial_transform& adj_a, spatial_vector& adj_s, const spatial_vector& adj_ret)
{
    printf("impl\n");
}

// should match model.py
#define JOINT_PRISMATIC 0
#define JOINT_REVOLUTE 1
#define JOINT_FIXED 2
#define JOINT_FREE 3


CUDA_CALLABLE inline spatial_transform spatial_jcalc(int type, float* joint_q, float3 axis, int start)
{
    if (type == JOINT_REVOLUTE)
    {
        float q = joint_q[start];
        spatial_transform X_jc = spatial_transform(float3(), quat_from_axis_angle(axis, q));
        return X_jc;
    }
    else if (type == JOINT_PRISMATIC)
    {
        float q = joint_q[start];
        spatial_transform X_jc = spatial_transform(axis*q, quat_identity());
        return X_jc;
    }
    else if (type == JOINT_FREE)
    {
        float px = joint_q[start+0];
        float py = joint_q[start+1];
        float pz = joint_q[start+2];
        
        float qx = joint_q[start+3];
        float qy = joint_q[start+4];
        float qz = joint_q[start+5];
        float qw = joint_q[start+6];
        
        spatial_transform X_jc = spatial_transform(float3(px, py, pz), quat(qx, qy, qz, qw));
        return X_jc;
    }

    // JOINT_FIXED
    return spatial_transform(float3(), quat_identity());
}

CUDA_CALLABLE inline void adj_spatial_jcalc(int type, float* q, float3 axis, int start, int& adj_type, float* adj_q, float3& adj_axis, int& adj_start, const spatial_transform& adj_ret)
{
    if (type == JOINT_REVOLUTE)
    {
        adj_quat_from_axis_angle(axis, q[start], adj_axis, adj_q[start], adj_ret.q);
    }
    else if (type == JOINT_PRISMATIC)
    {
        adj_mul(axis, q[start], adj_axis, adj_q[start], adj_ret.p);
    }
    else if (type == JOINT_FREE)
    {
        adj_q[start+0] += adj_ret.p.x;
        adj_q[start+1] += adj_ret.p.y;
        adj_q[start+2] += adj_ret.p.z;
        
        adj_q[start+3] += adj_ret.q.x;
        adj_q[start+4] += adj_ret.q.y;
        adj_q[start+5] += adj_ret.q.z;
        adj_q[start+6] += adj_ret.q.w;
    }
}

struct spatial_matrix
{
    float data[6][6] = {0};

    CUDA_CALLABLE inline spatial_matrix()
    {

    }
};



inline CUDA_CALLABLE float index(const spatial_matrix& m, int row, int col)
{
    return m.data[row][col];
}

inline CUDA_CALLABLE spatial_matrix spatial_transform_inertia(const spatial_transform& xform, const spatial_matrix& I)
{
    // I_a = np.dot(np.dot(spatial_adjoint_dual(t_ab), I_b), spatial_adjoint(transform_inverse(t_ab)))



    // calculate 
    spatial_matrix I_a;
    return I_a;
}


inline CUDA_CALLABLE spatial_vector mul(const spatial_matrix& a, const spatial_vector& b)
{
    // float3 r = a.get_col(0)*b.x +
    //            a.get_col(1)*b.y +
    //            a.get_col(2)*b.z;
    
    // return r;

    return spatial_vector(); // todo:
}

inline CUDA_CALLABLE void adj_mul(const spatial_matrix& a, const spatial_vector& b, spatial_matrix& adj_a, spatial_vector& adj_b, const spatial_vector& adj_ret)
{
    // todo:
    // adj_a += outer(adj_ret, b);
    // adj_b += mul(transpose(a), adj_ret);    
}




// // large matrix multiply routines
// void matrix_mutiply(const float* a, const float* b, float* c, int i, int j, int k)
// {
    
// }

// void matrix_solve(const float* matrix, int i, int j, float* vector)
// {
// }
