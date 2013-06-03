
kernel void Negate(global float* image, const unsigned long xMax, const unsigned long yMax, const unsigned long zMax)
{
    unsigned long index = (get_global_id(0) + get_global_id(1) * xMax) * zMax + get_global_id(2);
    if (get_global_id(0) >= xMax) return;
    if (get_global_id(1) >= yMax) return;
    if (get_global_id(2) >= zMax) return;

    image[index] = 255.0 - image[index];
}

kernel void Brightness(global float* image, const unsigned long xMax, const unsigned long yMax, const unsigned long zMax, const float brightness)
{
    unsigned long index = (get_global_id(0) + get_global_id(1) * xMax) * zMax + get_global_id(2);
    if (get_global_id(0) >= xMax) return;
    if (get_global_id(1) >= yMax) return;
    if (get_global_id(2) >= zMax) return;
    
    image[index] = brightness + image[index];
    if (image[index] > 255.0)
        image[index] = 255.0;
    if (image[index] < 0.0)
        image[index] = 0.0;
}

kernel void Threshold(global float* image, const unsigned long xMax, const unsigned long yMax, const unsigned long zMax, const float threshold)
{
    unsigned long index = (get_global_id(0) + get_global_id(1) * xMax) * zMax + get_global_id(2);
    if (get_global_id(0) >= xMax) return;
    if (get_global_id(1) >= yMax) return;
    if (get_global_id(2) >= zMax) return;

    if (image[index] < threshold)
        image[index] = 0.0;
    else
        image[index] = 255.0;
}

kernel void Contrast(global float* image, const unsigned long xMax, const unsigned long yMax, const unsigned long zMax, const float minI, const float maxI)
{
    unsigned long index = (get_global_id(0) + get_global_id(1) * xMax) * zMax + get_global_id(2);
    if (get_global_id(0) >= xMax) return;
    if (get_global_id(1) >= yMax) return;
    if (get_global_id(2) >= zMax) return;

    image[index] = (image[index] - minI) / (maxI - minI);
    if (image[index] > 255.0)
        image[index] = 255.0;
    if (image[index] < 0.0)
        image[index] = 0.0;
}

kernel void Smoothen(global float* image, const unsigned long xMax, const unsigned long yMax, const unsigned long zMax)
{
    unsigned long index = (get_global_id(0) + get_global_id(1) * xMax) * zMax + get_global_id(2);
    if (get_global_id(0) >= xMax) return;
    if (get_global_id(1) >= yMax) return;
    if (get_global_id(2) >= zMax) return;

    unsigned int count = 1;
    float value = image[index];
    if (get_global_id(0) != 0) {
        value += image[index - zMax];
        count++;
    }
    if (get_global_id(0) != xMax - 1) {
        value += image[index + zMax];
        count++;
    }
    if (get_global_id(1) != 0) {
        value += image[index - (zMax * xMax)];
        count++;
    }
    if (get_global_id(1) != yMax - 1) {
        value += image[index + (zMax * xMax)];
        count++;
    }

    image[index] = value / (float) count;
}
