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
__global__ void setup_kernel(float4* pos, float4* vel,unsigned int width, unsigned int height, unsigned int depth,float time)
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


		pos[(x*depth +y)*width+z] = make_float4( 1+u*4,1+v*4,1+w*4, 1 );
			
		vel[(x*depth +y)*width+z] = make_float4( 0, 0, 0, 0); 

		return;

}

///////////////////////////////////////////////////////////////////////////////
//! Simple kernel to modify vertex positions in sine wave pattern
//! @param data  data in global memory
///////////////////////////////////////////////////////////////////////////////
__global__ void kernel(float4* pos, float4* vel,unsigned int width, unsigned int height, unsigned int depth,float time, bool init)
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

		pos[(x*depth +y)*width+z] = make_float4( 1+u*4,1+v*4,1+w*4, 1 );
			
		vel[(x*depth +y)*width+z] = make_float4( 0, 0, 0, 0); 

		return;
	}
	else
	{
		float4 old_pos = pos[(x*depth +y)*width+z];
		float4 old_vel = vel[(x*depth +y)*width+z];

		float radius_squared = old_pos.x * old_pos.x + old_pos.y * old_pos.y + old_pos.z * old_pos.z;

		/*float4 accel = make_float4(-old_pos.x / (radius_squared),
								-old_pos.y / (radius_squared),
								-old_pos.z / (radius_squared),
								0);  
								*/
		float4 accel = make_float4(-old_pos.x / (radius_squared*1000000),
								-old_pos.y / (radius_squared*1000000),
								-old_pos.z / (radius_squared*1000000),
								0); 

	//	accel.x -= v*0.0001;// (0.001)*(cos(angle_rad+0.01)-cos(angle_rad));
	//	accel.y += u*0.0001;//(0.001)*(sin(angle_rad+0.01)-sin(angle_rad));

		//vel[y*width+x] = make_float4(  (v)*cos(angle_rad+0.001)-v*cos(angle_rad), (v)*sin(angle_rad+0.001) - v*sin(angle_rad), 0, 1); 

		float4 new_vel = make_float4(old_vel.x + accel.x*time,
									old_vel.y + accel.y*time,
									old_vel.z + accel.z*time,0);


		float4 new_pos = make_float4( old_pos.x + old_vel.x * time + 0.5 * accel.x*time*time,
								  old_pos.y + old_vel.y * time+ 0.5 * accel.y*time*time,
								  old_pos.z + old_vel.z * time+ 0.5 * accel.z*time*time,
								  1);
								  
/*		float4 new_pos = make_float4( old_pos.x + old_vel.x / 100000,
								  old_pos.y+ old_vel.y / 100000,
								  w,
								  1);*/
		pos[(x*depth +y)*width+z] = new_pos;
		vel[(x*depth +y)*width+z] = new_vel;
	}
}



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



		pos[(z*depth +y)*width+x] = make_float4( u/2,v/2,w/2, 1 );
			
		vel[(z*depth +y)*width+x] = make_float4( 0, 0, 0, 0); 

		return;
	}
	else
	{
		float4 old_pos = pos[(z*depth +y)*width+x];
		float4 old_vel = vel[(z*depth +y)*width+x];

		/* non random way of reinjecting particles that are out of "meaningful" range
		if( old_pos.x >= 5 || old_pos.y >= 5 || old_pos.z >= 5 )
		{
			float4 new_pos = make_float4(0.0f, 0.5f, 2.0f, 0.0f );
			float4 new_vel = make_float4(0.0002f, 0.0f, 0.0002f, 0.0f);
			pos[(z*depth +y)*width+x] = new_pos;
			vel[(z*depth +y)*width+x] = new_vel;
		}
		else*/
		if( old_pos.x >= 5 || old_pos.y >= 5 || old_pos.z >= 5 )
		{
			float4 new_pos = make_float4(u, v, w, 0.0f );
			float4 new_vel = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
			pos[(z*depth +y)*width+x] = new_pos;
			vel[(z*depth +y)*width+x] = new_vel;
		}
		else
		{

			float radius_squared = old_pos.x * old_pos.x + old_pos.y * old_pos.y + old_pos.z * old_pos.z;

			/*float4 accel = make_float4(-old_pos.x / (radius_squared),
									-old_pos.y / (radius_squared),
									-old_pos.z / (radius_squared),
									0);  
									*/
			// electric field
			float constant = 1/1000000;
			float4 accel = make_float4(-old_pos.x / (radius_squared*10000000),
									-old_pos.y / (radius_squared*10000000),
									-old_pos.z / (radius_squared*10000000),
									0); 
			// magnetic field
			// F = [constant 1] scalar(q1 * q2 /r^2) * vector( v1 X (v2 X (pq1-pq2)))
			// the equation can be rephrased as
			// a = [constant] * (v1 X (v2 X (pq1-pq2)))/ r ^ 2
			// with [constant] as [constant 1] * q1*q2
			// and [constant 1] from the above equation is 1/ 1000000
			// now the field is *generated* by a moving particle and stationary attractors
			// would not generate any kind of field.
			// To model the field of a planet we can best approximate by a circular current carrying loop,
			// it should be obvious that a complete field should be given by the integral of all loops through
			// the sphere, however in practice such an integral is probably impossible, the field could be modelled
			// by discrete loops. 

			// or it can be modelled like a bar magnet through the center. This can be treated as 2 attractive fields
			// 1 positive and 1 negative...

			// Dipole
			// for a dipole of 2 point charges of charge +Q and -Q seperated by a distance d on the z axis
			// the net Potential generated by these point charges is just the sum of the 2 point charges
			// given V = potential, v1 = [vector from attractor +Q to particle]
			//					    v2 = [vector from attractor -Q to particle]
			// [constant 3] = magnetic field rescaling 
			// r1 = length(v1)
			// r2 = length(v2)
		
			// V = [constant 3] * Q / r1 - [constant 3] * Q / r1
			// V = [constant 3] * Q * (r2-r1)/(r2*r1)

			float4 Q1_pos = make_float4( 0.0f, 0.1f, 0.0f, 0.0f ); // say 10 units upwards

			float4 Q2_pos = make_float4( 0.0f, -0.1f, 0.0f, 0.0f ); // 10 units downwards

			float4 Q1_to_part = make_float4( old_pos.x - Q1_pos.x,
											  old_pos.y - Q1_pos.y,
											  old_pos.z - Q1_pos.z, 0.0f);

			float4 Q2_to_part = make_float4( old_pos.x - Q2_pos.x,
											  old_pos.y - Q2_pos.y,
											  old_pos.z - Q2_pos.z, 0.0f);

			// now these vectors are important, however the need to take the gradient of the potential to further
			// improve this forumla is inconvenient, so I am going to computer the attractors using the same forumla
			// as above, constant * Q / r^2 for each charge, taking into account the signs of the charge.
			// how this is reinterpreted as a magnetic field will be shown below.

			float r1_squared = ( Q1_to_part.x*Q1_to_part.x + Q1_to_part.y*Q1_to_part.y + Q1_to_part.z*Q1_to_part.z);
			float r2_squared = ( Q2_to_part.x*Q2_to_part.x + Q2_to_part.y*Q2_to_part.y + Q2_to_part.z*Q2_to_part.z);

			constant =1/100000000000; // changing the constant

			// now I need the lengths of the lines between 
			float4 accel_pre_mag_positive = make_float4(-(Q1_to_part.x) / (r1_squared*1000),
														-(Q1_to_part.y) / (r1_squared*1000),
														-(Q1_to_part.z) / (r1_squared*1000),
														0); 

			float4 accel_pre_mag_negative = make_float4((Q2_to_part.x) / (r2_squared*1000),
														(Q2_to_part.y) / (r2_squared*1000),
														(Q2_to_part.z) / (r2_squared*1000),
														0); 

			// summing the 2 fields
			float4 accel_pre_mag = make_float4( accel_pre_mag_positive.x + accel_pre_mag_negative.x,
												accel_pre_mag_positive.y + accel_pre_mag_negative.y,
												accel_pre_mag_positive.z + accel_pre_mag_negative.z, 0.0f);

			// now that is the field force calculated for the dipole. To mix the magnetic force with the electric
			// force, we need to use the *Lorentz Force Law* equation
			// F = qE + q(v X B)
			// now assuming B[scaled] is given by accel_pre_mag, we need
			// current_veloctiy [cross] B
		
			// using the cross product notation, 
			// |i j k
			// |a1 a2 a3
			// |b1 b2 b3
			// 
			float i1 = old_vel.y * accel_pre_mag.z - old_vel.z * accel_pre_mag.y;
			float j1 = old_vel.x * accel_pre_mag.z - old_vel.z * accel_pre_mag.x;
			float k1 = old_vel.x * accel_pre_mag.y - old_vel.y * accel_pre_mag.x;
			//float4 accel = make_float4(accel_pre_mag.x,accel_pre_mag.y,accel_pre_mag.z,1.0f);	
			// now add this to accel to complete the lorentz force...
			accel.x += i1;
			accel.y += j1;
			accel.z += k1;
			//accel.x += 0.01*accel_pre_mag.x;
			//accel.y += 0.01*accel_pre_mag.y;
			//accel.z += 0.01*accel_pre_mag.z;
								
			float4 new_vel = make_float4(old_vel.x + accel.x*time,
										old_vel.y + accel.y*time,
										old_vel.z + accel.z*time,0);


			float4 new_pos = make_float4( old_pos.x + old_vel.x * time + 0.5 * accel.x*time*time,
									      old_pos.y + old_vel.y * time+ 0.5 * accel.y*time*time,
									      old_pos.z + old_vel.z * time+ 0.5 * accel.z*time*time,
									      1);
								  
	/*		float4 new_pos = make_float4( old_pos.x + old_vel.x / 100000,
									  old_pos.y+ old_vel.y / 100000,
									  w,
									  1);*/
			pos[(z*depth +y)*width+x] = new_pos;
			vel[(z*depth +y)*width+x] = new_vel;
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

extern "C" void call_setup_kernel(float4* pos, float4* vel,unsigned int width, unsigned int height, unsigned int depth,float time)
{
    // execute the kernel
    dim3 block(8, 8, 8);
    dim3 grid(width / block.x, height / block.y, depth/ block.z);
    //kernel<<< grid, block>>>(pos, vel,mesh_width, mesh_height,depth, time, init);
	setup_kernel<<< grid, block>>>(pos, vel,width, height,depth, time);
}


#endif // #ifndef _SIMPLEGL_KERNEL_H_
