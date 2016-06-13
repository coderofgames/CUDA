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

// true if in front
__device__ bool IsPointInFrontOfPlane(float4 point, float4 plane, float eps)
{
	bool ret = false;

	float d = point.x*plane.x + point.y*plane.y + point.z*plane.z + plane.w;
	
	if ( (d + eps) > 0 ) 
	{
		ret = true;
	}

	return ret;
}
 

__device__ bool IsPointInSphere(float4 point, float4 center, float radius, float eps)
{
	bool ret = false;

	float4 c_to_p = make_float4( point.x - center.x,
										  point.y - center.y,
										  point.z - center.z,
										  0.0f);
	float d = sqrt( c_to_p.x * c_to_p.x + c_to_p.y * c_to_p.y + c_to_p.z * c_to_p.z );
	
	if ( (d + eps) < radius ) 
	{
		ret = true;
	}

	return ret;
}

// evalue y=x*x-z*z
__device__ bool IsPointAboveMonkeySaddleY(float4 point, float4 origin, float height, float eps)
{
	bool ret = false;

	// assume it fills the space for now
	float h = point.x * point.x - point.y*point.y;

	if ( h < point.z ) 
	{
		ret = true;
	}

	return ret;
}

///////////////////////////////////////////////////////////////////////////////
//! Simple kernel to modify vertex positions in sine wave pattern
//! @param data  data in global memory
///////////////////////////////////////////////////////////////////////////////
__global__ void kernel(float4* pos, float4* vel,unsigned int width, unsigned int height, float time, bool init, float4 plane,float elevation, bool bWavy)
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
		if( bWavy )
		{
			pos[y*width+x] = make_float4( u, v, w + elevation, 1 );
		}
		else
		{
			pos[y*width+x] = make_float4( u, v, elevation, 1 );
		}
			
		vel[y*width+x] = make_float4( 0, 255.0f, 0, 1); 

		return;
	}
	else
	{
		float4 old_pos = pos[y*width+x];
		float4 old_vel = vel[y*width+x];

		if( IsPointInFrontOfPlane( old_pos, plane, 0.000001 ))
		{
			//pos[y*width+x] = new_pos;
			
			
			if( old_vel.x != 255.0f ) // point entered, add some blue
			{
				vel[y*width+x] = make_float4(255.0f,0.0f,255.0f,1.0f);	
			}	
			else
			{

				
				vel[y*width+x] = make_float4(255.0f,0.0f,0.0f,1.0f);	
				
			}
		}
		else
		{
			vel[y*width+x] = make_float4(0.0f,255.0f,0.0f,1.0f);	
		}


		if( bWavy )
		{
			pos[y*width+x] = make_float4( u, v, w + elevation, 1 );
		}
		else
		{
			pos[y*width+x] = make_float4( u, v, elevation, 1 );
		}

		return;
	}

	return;
}


///////////////////////////////////////////////////////////////////////////////
//! Simple kernel to modify vertex positions in sine wave pattern
//! @param data  data in global memory
///////////////////////////////////////////////////////////////////////////////
__global__ void kernel_clip_sphere(float4* pos, float4* vel,unsigned int width, unsigned int height, float time, bool init, float4 center, float radius, float elevation, bool bWavy)
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
		if( bWavy )
		{
			pos[y*width+x] = make_float4( u, v, w + elevation, 1 );
		}
		else
		{
			pos[y*width+x] = make_float4( u, v, elevation, 1 );
		}	
			
		vel[y*width+x] = make_float4( 0, 255.0f, 0, 1); 

		return;
	}
	else
	{
		float4 old_pos = pos[y*width+x];
		float4 old_vel = vel[y*width+x];

		if( IsPointInSphere( old_pos, center, radius, 0.000001 ))
		{
			//pos[y*width+x] = new_pos;
			
			
			if( old_vel.x != 255.0f ) // point entered, add some blue
			{
				vel[y*width+x] = make_float4(255.0f,0.0f,255.0f,1.0f);	
			}	
			else
			{

				
				vel[y*width+x] = make_float4(255.0f,0.0f,0.0f,1.0f);	
				
			}
		}
		else
		{
			vel[y*width+x] = make_float4(0.0f,255.0f,0.0f,0.1f);	
		}


		if( bWavy )
		{
			pos[y*width+x] = make_float4( u, v, w + elevation, 1 );
		}
		else
		{
			pos[y*width+x] = make_float4( u, v, elevation, 1 );
		}

		return;
	}

	return;
}


///////////////////////////////////////////////////////////////////////////////
//! Simple kernel to modify vertex positions in sine wave pattern
//! @param data  data in global memory
///////////////////////////////////////////////////////////////////////////////
__global__ void kernel_clip_surface_Y_saddle(float4* pos, float4* vel,unsigned int width, 
											 unsigned int height, float time, bool init, 
											 float4 origin, float elevation, bool bWavy)
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
		if( bWavy )
		{
			pos[y*width+x] = make_float4( u, v, w + elevation, 1 );
		}
		else
		{
			pos[y*width+x] = make_float4( u, v, elevation, 1 );
		}	
			
		vel[y*width+x] = make_float4( 0, 255.0f, 0, 1); 

		return;
	}
	else
	{
		float4 old_pos = pos[y*width+x];
		float4 old_vel = vel[y*width+x];

		if( IsPointAboveMonkeySaddleY( old_pos, origin, elevation, 0.000001 ))
		{
			//pos[y*width+x] = new_pos;
			
			
			if( old_vel.x != 255.0f ) // point entered, add some blue
			{
				vel[y*width+x] = make_float4(255.0f,0.0f,255.0f,1.0f);	
			}	
			else
			{

				
				vel[y*width+x] = make_float4(255.0f,0.0f,0.0f,1.0f);	
				
			}
		}
		else
		{
			vel[y*width+x] = make_float4(0.0f,255.0f,0.0f,0.1f);	
		}


		if( bWavy )
		{
			pos[y*width+x] = make_float4( u, v, w + elevation, 1 );
		}
		else
		{
			pos[y*width+x] = make_float4( u, v, elevation, 1 );
		}

		return;
	}

	return;
}


// Wrapper for the __global__ call that sets up the kernel call
extern "C" void launch_kernel(float4* pos, float4* vel, 
								unsigned int mesh_width, unsigned int mesh_height, 
								float time, bool init, float4 plane, float elevation, bool bWavy)
{
    // execute the kernel
    dim3 block(8, 8, 1);
    dim3 grid(mesh_width / block.x, mesh_height / block.y, 1);
    kernel<<< grid, block>>>(pos, vel, mesh_width, mesh_height, 
							 time, init, plane, elevation, bWavy);
}



// Wrapper for the __global__ call that sets up the kernel call
extern "C" void launch_clip_sphere(float4* pos, float4* vel, 
									unsigned int mesh_width, unsigned int mesh_height, 
									float time, bool init, float4 center, 
									float radius, float elevation, bool bWavy)
{
    // execute the kernel
    dim3 block(8, 8, 1);
    dim3 grid(mesh_width / block.x, mesh_height / block.y, 1);
    kernel_clip_sphere<<< grid, block>>>( pos, vel,mesh_width, mesh_height, time, 
										  init, center, radius, elevation, bWavy);
}


// Wrapper for the __global__ call that sets up the kernel call
extern "C" void launch_clip_surface_Y_saddle(float4* pos, float4* vel, 
											unsigned int mesh_width, unsigned int mesh_height, 
											float time, bool init, float4 origin, 
											float elevation, bool bWavy)
{
    // execute the kernel
    dim3 block(8, 8, 1);
    dim3 grid(mesh_width / block.x, mesh_height / block.y, 1);
    kernel_clip_surface_Y_saddle<<< grid, block>>>(pos, vel, mesh_width, mesh_height, 
												   time, init, origin, elevation, bWavy);

}
#endif // #ifndef _SIMPLEGL_KERNEL_H_
