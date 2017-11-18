__kernel void vec_add( __global float* a, __global float* b, __global float* c, const unsigned int count) {

    int i = get_global_id(0);
    c[i] = a[i] + b[i];
}