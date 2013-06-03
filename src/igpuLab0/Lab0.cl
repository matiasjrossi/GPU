
    // OpenCL Kernel Function for element by element 
    kernel void FillVector(global float* a, int numElements) 
{
        // get index into global data array
        int iGID = get_global_id(0);
        
        // bound check (equivalent to the limit on a 'for' loop for standard/serial C code
        if (iGID >= numElements)  
        {
            return;
        }

        // set 1
        a[iGID] =a[iGID]+ 1;
    }
