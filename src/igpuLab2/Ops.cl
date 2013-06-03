kernel void AddOne(global const float* a, global float* b, int numElements) 
{
    int iGID = get_global_id(0);

    if (iGID >= numElements)  {
        return;
    }

    b[iGID] = a[iGID] + 1;
}

kernel void Mixed1(global const float* a, global float* b, int numElements) 
{
    int iGID = get_global_id(0);

    if (iGID >= numElements)  {
        return;
    }

    b[iGID] = pow(a[iGID], 5) * sin(a[iGID]) * sqrt(a[iGID]);
}

kernel void Mixed2(global const float* a, global float* b, int numElements) 
{
    int iGID = get_global_id(0);

    if (iGID >= numElements)  {
        return;
    }

    b[iGID] = a[iGID-1] + a[iGID+1];
}