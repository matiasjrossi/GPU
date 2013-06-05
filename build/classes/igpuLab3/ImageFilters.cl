
kernel void Negate(global float* image, const unsigned long numElements)
{
    unsigned long index = get_global_id(0);
    if (index >= numElements) return;

    image[index] = 255.0 - image[index];
}

kernel void Brightness(global float* image, const unsigned long numElements, const float brightness)
{
    unsigned long index = get_global_id(0);
    if (index >= numElements) return;
    
    image[index] = brightness + image[index];
    if (image[index] > 255.0)
        image[index] = 255.0;
    if (image[index] < 0.0)
        image[index] = 0.0;
}

kernel void Threshold(global float* image, const unsigned long numElements, const float threshold)
{
    unsigned long index = get_global_id(0);
    if (index >= numElements) return;

    if (image[index] < threshold)
        image[index] = 0.0;
    else
        image[index] = 255.0;
}

kernel void Contrast(global float* image, const unsigned long numElements, const float minI, const float maxI)
{
    unsigned long index = get_global_id(0);
    if (index >= numElements) return;
    
    image[index] = (image[index] - minI) / (maxI - minI);
    if (image[index] > 255.0)
        image[index] = 255.0;
    if (image[index] < 0.0)
        image[index] = 0.0;
}

kernel void Smoothen(global float* image, const unsigned long numElements, const unsigned long width)
{
    unsigned long index = get_global_id(0);
    if (index >= numElements) return;

    unsigned long cellWidth = 3;
    unsigned long rowWidth = width * cellWidth;
    
    unsigned int count = 1;
    float value = image[index];
    
    if ((index - cellWidth >= 0) && (index - cellWidth < numElements)) {
        count++;
        value += image[index - cellWidth];
    }
    if ((index + cellWidth >= 0) && (index + cellWidth < numElements)) {
        count++;
        value += image[index + cellWidth];
    }
    if ((index - rowWidth >= 0) && (index - rowWidth < numElements)) {
        count++;
        value += image[index - rowWidth];
    }
    if ((index + rowWidth >= 0) && (index + rowWidth < numElements)) {
        count++;
        value += image[index + rowWidth];
    }

    image[index] = value / as_float(count);
}
