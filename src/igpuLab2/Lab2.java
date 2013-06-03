package igpuLab2;

import com.jogamp.opencl.CLBuffer;
import com.jogamp.opencl.CLCommandQueue;
import com.jogamp.opencl.CLContext;
import com.jogamp.opencl.CLDevice;
import com.jogamp.opencl.CLKernel;
import com.jogamp.opencl.CLProgram;
import java.io.IOException;
import java.nio.FloatBuffer;
import java.util.Random;

import static java.lang.System.*;
import static com.jogamp.opencl.CLMemory.Mem.*;
import static java.lang.Math.*;


public class Lab2 {

    static final int SEED_A = 12345;
    static final int[] DATA_SIZES = {10485760 /*10MB*/, 52428800 /*50MB*/, 104857600 /*100MB*/};
    static final int[] MAX_LOCAL_WORK_SIZE = {2, 16, 64, 128, 512};
    static final String[] KERNEL_NAMES = {"AddOne", "Mixed1", "Mixed2"};
    
    static CLContext context;
    static CLDevice device;
    static CLCommandQueue queue;

    public static void main(String[] args) throws IOException {

        // create OpenCL context
        context = CLContext.create();
        out.println("Context created: " + context);

        try {
            // select fastest device
            device = context.getMaxFlopsDevice();
            out.println("Using device: " + device);

            // create command queue on device
            queue = device.createCommandQueue();

            // load CL program and build it
            CLProgram program = context.createProgram(Lab2.class.getResourceAsStream("Ops.cl"));
            program.build();

            out.println();
            
            for (String kernelName: KERNEL_NAMES)
                for (int dataSize: DATA_SIZES)
                    for (int maxLocalWorkSize: MAX_LOCAL_WORK_SIZE) {
                        out.println(
                                "Measuring " + kernelName +
                                " with maxLocalWorkSize=" + maxLocalWorkSize +
                                " and dataSize=" + dataSize);
                        processCL(program.createCLKernel(kernelName), maxLocalWorkSize, dataSize);
                    }


        } finally {
            // cleanup all resources associated with this context
            context.release();
        }

    }

    private static void processCL(CLKernel kernel, int maxLocalWorkSize, int dataSize) {

        int elementCount = dataSize / (Float.SIZE / 8);
        
        int localWorkSize = min((int) kernel.getWorkGroupSize(device), maxLocalWorkSize);
        int globalWorkSize = roundUp(localWorkSize, elementCount);

        out.println("With localWorkSize=" + localWorkSize + " globalWorkSize=" + globalWorkSize);


        // A, B are input buffers, C is for the result
        CLBuffer<FloatBuffer> clBufferA = context.createFloatBuffer(elementCount, READ_ONLY);
        CLBuffer<FloatBuffer> clBufferB = context.createFloatBuffer(elementCount, WRITE_ONLY);

        out.println(
                "Allocated device memory: "
                + (clBufferA.getCLSize() + clBufferB.getCLSize()) / 1000000
                + "MB");

        // fill input buffers with random numbers
        fillBuffer(clBufferA.getBuffer(), SEED_A);

        // copy data to GPU
        long inputTime = nanoTime();
        queue.putWriteBuffer(clBufferA, true);
        inputTime = nanoTime() - inputTime;
        
        // call GPU
        kernel.setArg(0, clBufferA);
        kernel.setArg(1, clBufferB);
        kernel.setArg(2, elementCount);

        long computeTime = nanoTime();
        queue.put1DRangeKernel(kernel, 0, globalWorkSize, localWorkSize);
        queue.finish();
        computeTime = nanoTime() - computeTime;


        // get results
        long outputTime = nanoTime();
        queue.putReadBuffer(clBufferB, true);
        outputTime = nanoTime() - outputTime;

        out.println("Input time: " + inputTime/1000000.0 + "ms");
        out.println("Compute time: " + computeTime/1000000.0 + "ms");
        out.println("Output time: " + outputTime/1000000.0 + "ms");
        out.println("TOTAL TIME: " + (inputTime + computeTime + outputTime)/1000000.0 + "ms");
                
        out.println();
        
        clBufferA.release();
        clBufferB.release();
        kernel.release();
    }

    private static void fillBuffer(FloatBuffer buffer, int seed) {
        Random rnd = new Random(seed);
        while (buffer.remaining() != 0) {
            buffer.put(rnd.nextFloat() * 100);
        }
        buffer.rewind();
    }

    private static int roundUp(int groupSize, int globalSize) {
        int r = globalSize % groupSize;
        if (r == 0) {
            return globalSize;
        } else {
            return globalSize + groupSize - r;
        }
    }
}