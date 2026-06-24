__global__ void vector_mul(
    const int* a,
    const int* b,
    int* c,
    int numElements)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    int numVec4 = numElements / 4;

    const int4* a4 = reinterpret_cast<const int4*>(a);
    const int4* b4 = reinterpret_cast<const int4*>(b);
    int4* c4 = reinterpret_cast<int4*>(c);

    int stride = blockDim.x * gridDim.x;

    // Process int4 chunks
    for (int vecIdx = tid; vecIdx < numVec4; vecIdx += stride) {
        int4 av = a4[vecIdx];
        int4 bv = b4[vecIdx];

        int4 cv;
        cv.x = av.x * bv.x;
        cv.y = av.y * bv.y;
        cv.z = av.z * bv.z;
        cv.w = av.w * bv.w;

        c4[vecIdx] = cv;
    }

    // Handle remaining elements
    int tailStart = numVec4 * 4;

    for (int idx = tailStart + tid;
         idx < numElements;
         idx += stride)
    {
        c[idx] = a[idx] * b[idx];
    }
}
