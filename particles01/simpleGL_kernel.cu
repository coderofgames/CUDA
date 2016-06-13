/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

 /* This example demonstrates how to use the Cuda OpenGL bindings with the
  * runtime API.
  * Device code.
  */

  //

#ifndef _SIMPLEGL_KERNEL_H_
#define _SIMPLEGL_KERNEL_H_



///////////////////////////////////////////////////////////////////////////////
//! Simple kernel to modify vertex positions in sine wave pattern
//! @param data  data in global memory
///////////////////////////////////////////////////////////////////////////////
__global__ void kernel(float4* pos, float4* vel,unsigned int width, unsigned int height, float time, bool init)
{
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;


		// calculate uv coordinates
		float u = x / (float) width;
		float v = y / (float) height;
		u = u*2.0f - 1.0f;
		v = v*2.0f - 1.0f;

		// calculate simple sine wave pattern
		float freq = 4.0f;
		float w = sinf(u*freq + time) * cosf(v*freq + time) * 0.5f;

	if( init )
	{


		pos[y*width+x] = make_float4( u, v, w, 1 );

		/*float radius_squared = u * u + v * v + w * w;

		float4 accel = make_float4(u*10 / radius_squared,
								v*10 / radius_squared,
								w*10 / radius_squared,
								0);*/
								  
		vel[y*width+x] = make_float4( 0, 0, 0, 1); 

		return;
	}
	else
	{
		float4 old_pos = pos[y*width+x];
		float4 old_vel = vel[y*width+x];

		float radius_squared = old_pos.x * old_pos.x + old_pos.y * old_pos.y + old_pos.z * old_pos.z;

		// note - this is fudged and simply uses the position, not the normal
		float4 accel = make_float4(-old_pos.x / (radius_squared),
								-old_pos.y / (radius_squared),
								-old_pos.z / (radius_squared),
								0);  

		/*float4 accel = make_float4(-old_pos.x / (radius_squared*10000000),
								-old_pos.y / (radius_squared*10000000),
								-old_pos.z / (radius_squared*10000000),
								0); 
								*/
		float4 new_vel = make_float4(old_vel.x + accel.x*time,
									old_vel.y + accel.y*time,
									old_vel.z + accel.z*time,0);

/*
		float4 new_pos = make_float4( old_pos.x + old_vel.x * time + 0.5 * accel.x*time*time,
								  old_pos.y + old_vel.y * time+ 0.5 * accel.y*time*time,
								  old_pos.z + old_vel.z * time+ 0.5 * accel.z*time*time,
								  1);
								  */
		float4 new_pos = make_float4( old_pos.x + old_vel.x / 100000,
								  old_pos.y+ old_vel.y / 100000,
								  w,
								  1);
		pos[y*width+x] = new_pos;
		vel[y*width+x] = new_vel;
	}
}

// Wrapper for the __global__ call that sets up the kernel call
extern "C" void launch_kernel(float4* pos, float4* vel, unsigned int mesh_width, unsigned int mesh_height, float time, bool init)
{
    // execute the kernel
    dim3 block(8, 8, 1);
    dim3 grid(mesh_width / block.x, mesh_height / block.y, 1);
    kernel<<< grid, block>>>(pos, vel,mesh_width, mesh_height, time, init);
}

#endif // #ifndef _SIMPLEGL_KERNEL_H_
