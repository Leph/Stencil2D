
__kernel void
copy(__global float *B,
        __global float *A,
        unsigned int line_size)
{
   const unsigned int x = get_global_id(0);
   const unsigned int y = get_global_id(1);

   A += line_size + 16; // OFFSET
   B += line_size + 16; // OFFSET

   B[y * line_size + x] = A[y * line_size + x];
}
