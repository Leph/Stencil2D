
__kernel void
stencil(__global float *B,
        __global float *A,
        unsigned int line_size)
{
   unsigned int x = get_global_id(0);
   unsigned int y = get_global_id(1);

   A += line_size + 16; // OFFSET
   B += line_size + 16; // OFFSET

   for(int k=0; k<4; k++)
     B[(y*4 + k)*line_size + x] = 0.75 * A[(y*4 + k)*line_size + x ] +
                                  0.25*( A[(y*4 + k)*line_size + x - 1 ] +
				         A[(y*4 + k)*line_size + x + 1] +
                                         A[(y*4 + k - 1)*line_size + x ] +
					 A[(y*4 + k + 1)*line_size + x ] );
}
