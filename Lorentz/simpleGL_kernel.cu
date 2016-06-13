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


///////////////////////////////////////////////////////////////////////////////
//! Simple kernel to modify vertex positions in sine wave pattern
//! @param data  data in global memory
///////////////////////////////////////////////////////////////////////////////
__global__ void kernel_EM(float4* pos, float4* vel,unsigned int width, unsigned int height, unsigned int depth,float time, bool init)
{
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
	unsigned int z = blockIdx.z*blockDim.z + threadIdx.z;

		// calculate uv coordinates
		float u = x / (float) width;
		float v = y / (float) height;
		
		u = u*2.0f - 1.0f;
		v = v*2.0f - 1.0f;

		// calculate simple sine wave pattern
		float freq = 4.0f;
		float w = (z / (float) depth); //sinf(u*freq + time) * cosf(v*freq + time) * 0.5f;
		w = w*2.0f-1.0f;
			float angle_rad = 2*3.14159*u;
		float radius = v;

	if( init )
	{

	//	float theta = acos(u);
	//	float phi = asin(v);

		float dist = rsqrt(u*u+v*v);
		float distU=  u*dist;
		float distV= v*dist;
		
		float uvx = -v*0.00013;
		float uvy = -u*0.00013;
		float uvz = -0.002f;
		
		/*float4 cross_radius_up = make_float4( uvx,
											  uvy,
											  uvz, 
											  1.0f );
											  */

		float angle_rad = 2*3.14159*u;
		float radius = v;
	
	// stupid not a solution to this system
	// dependant on the last step.


		
		pos[(x*depth +y)*width+z] = make_float4(1.0f, 0.0f,0.0f, 1.0f);
			
		vel[(z*depth +y)*width+x] = make_float4( 0, 0, 0, 0); 

		return;
	}
	else
	{
		float4 old_pos = pos[(x*depth +y)*width+z];
		float4 old_vel = vel[(x*depth +y)*width+z];

		/*if( old_pos.x >= 5 || old_pos.y >= 5 || old_pos.z >= 5 )
		{
			float4 new_pos = make_float4(0.0f, 0.5f, 2.0f, 0.0f );
			float4 new_vel = make_float4(0.0002f, 0.0f, 0.0002f, 0.0f);
			pos[(x*depth +y)*width+z] = new_pos;
			vel[(x*depth +y)*width+z] = new_vel;
		}
		else*/
		{
			float threadIdxNew = (x*depth +y)*width+z;
			// Lorentz Attractor * 0.00000001
			float4 L_A = make_float4( old_pos.x +(10*(old_pos.y-old_pos.x)) *0.00000001* time*threadIdxNew, 
									old_pos.y+(-old_pos.x*old_pos.z +26.5*old_pos.x-old_pos.y)*0.00000001* time*threadIdxNew, 
									old_pos.z+(old_pos.x*old_pos.y-8*old_pos.z/3)*0.00000001* time*threadIdxNew, 0.0f);
				
			// velocity is not really computed here				
			float4 new_vel = make_float4(old_vel.x + L_A.x*time,
										old_vel.y + L_A.y*time,
										old_vel.z + L_A.z*time,0);

				  
			float4 new_pos = make_float4( L_A.x, L_A.y, L_A.z, 1.0f);//

			pos[(x*depth +y)*width+z] = new_pos;
			vel[(x*depth +y)*width+z] = new_vel;
		}
	}
}

// Wrapper for the __global__ call that sets up the kernel call
extern "C" void launch_kernel(float4* pos, float4* vel, unsigned int mesh_width, unsigned int mesh_height, unsigned int depth,float time, bool init)
{
    // execute the kernel
    dim3 block(8, 8, 8);
    dim3 grid(mesh_width / block.x, mesh_height / block.y, depth/ block.z);
    //kernel<<< grid, block>>>(pos, vel,mesh_width, mesh_height,depth, time, init);
	kernel_EM<<< grid, block>>>(pos, vel,mesh_width, mesh_height,depth, time, init);
}



#endif // #ifndef _SIMPLEGL_KERNEL_H_
