
kernel void AddVector(global const float* a, global const float* b, global float* c, int numElements) 
{
    // get index into global data array
    int iGID = get_global_id(0);

    // bound check (equivalent to the limit on a 'for' loop for standard/serial C code
    if (iGID >= numElements)  {
        return;
    }

    // add the vector elements
    c[iGID] = a[iGID] + b[iGID];
}

kernel void AddFloat(global const float* a, const float b, global float* c, int numElements) 
{
    // get index into global data array
    int iGID = get_global_id(0);

    // bound check (equivalent to the limit on a 'for' loop for standard/serial C code
    if (iGID >= numElements)  {
        return;
    }

    // add the float to the vector elements
    c[iGID] = a[iGID] + b;
}

kernel void Reverse(global const float* a, global float* b, int numElements) 
{
    // get index into global data array
    int iGID = get_global_id(0);

    // bound check (equivalent to the limit on a 'for' loop for standard/serial C code
    if (iGID >= numElements)  {
        return;
    }

    // copy the vector element
    b[iGID] = a[numElements - 1 - iGID];
}