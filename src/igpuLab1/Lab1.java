package igpuLab1;

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

public class Lab1 {
    
    static final int ELEMENT_COUNT = 14444771;
    static final int MAX_WORK_SIZE = 512;
    static final int SEED_A = 12345;
    static final int SEED_B = 67890;
    static final float FLOAT_TO_ADD = (float) 3.3934803;

    public static void main(String[] args) throws IOException {

        // create OpenCL context
        CLContext context = CLContext.create();
        out.println("Context created: " + context);
        
        try{
            // select fastest device
            CLDevice device = context.getMaxFlopsDevice();
            out.println("Using device: " + device);

            // create command queue on device
            CLCommandQueue queue = device.createCommandQueue();

            // load CL program and build it
            CLProgram program = context.createProgram(Lab1.class.getResourceAsStream("VectorOps.cl"));
            program.build();
            
            out.println();
            
            
/* AddVector */           
            // create CL kernel based on the previously built program
            CLKernel kernel = program.createCLKernel("AddVector");

            int localWorkSize = min((int)kernel.getWorkGroupSize(device), MAX_WORK_SIZE);
            int globalWorkSize = roundUp(localWorkSize, ELEMENT_COUNT);
            
            out.println("With localWorkSize=" + localWorkSize + " globalWorkSize=" + globalWorkSize);

            // A, B are input buffers, C is for the result
            CLBuffer<FloatBuffer> clBufferA = context.createFloatBuffer(ELEMENT_COUNT, READ_ONLY);
            CLBuffer<FloatBuffer> clBufferB = context.createFloatBuffer(ELEMENT_COUNT, READ_ONLY);
            CLBuffer<FloatBuffer> clBufferC = context.createFloatBuffer(ELEMENT_COUNT, WRITE_ONLY);
            
            out.println(
                    "Allocated device memory: " +
                    (clBufferA.getCLSize()+clBufferB.getCLSize()+clBufferC.getCLSize())/1000000 +
                    "MB");

            // fill input buffers with random numbers
            fillBuffer(clBufferA.getBuffer(), SEED_A);
            fillBuffer(clBufferB.getBuffer(), SEED_B);                       
            
            // copy data to GPU
            queue.putWriteBuffer(clBufferA, false);
            queue.putWriteBuffer(clBufferB, true);
            
            // call GPU
            kernel.setArg(0, clBufferA);
            kernel.setArg(1, clBufferB);
            kernel.setArg(2, clBufferC);
            kernel.setArg(3, ELEMENT_COUNT);
            
            long time = nanoTime();
            queue.put1DRangeKernel(kernel, 0, globalWorkSize, localWorkSize);
            queue.finish();
            time = nanoTime() - time;

            out.println("OCL AddVector took " + (time/1000000) + "ms");
            
            // get results
            queue.putReadBuffer(clBufferC, true);
            FloatBuffer seqBufferA = FloatBuffer.allocate(ELEMENT_COUNT);
            FloatBuffer seqBufferB = FloatBuffer.allocate(ELEMENT_COUNT);
            FloatBuffer seqBufferC = FloatBuffer.allocate(ELEMENT_COUNT);

            fillBuffer(seqBufferA, SEED_A);
            fillBuffer(seqBufferB, SEED_B);
            
            time = nanoTime();
            for (int i=0; i<ELEMENT_COUNT; ++i)
                seqBufferC.put(i, seqBufferA.get(i) + seqBufferB.get(i));
            time = nanoTime() - time;
            
            out.println("Sequential AddVector took " + (time/1000000) + "ms");
            
            out.print("Verifying results... ");
            boolean passed = true;
            for (int i=0; i<ELEMENT_COUNT; ++i)
                if (seqBufferC.get(i) != clBufferC.getBuffer().get(i)) {
                    passed = false;
                    break;
                }
            
            if (passed)
                out.println("OK");
            else
                out.println("FAILED");
            
            out.println();
            
            clBufferA.release();
            clBufferB.release();
            clBufferC.release();
            
            kernel.release();
            
/* AddReal */
            // create CL kernel based on the previously built program
            kernel = program.createCLKernel("AddFloat");

            localWorkSize = min((int)kernel.getWorkGroupSize(device), MAX_WORK_SIZE);
            globalWorkSize = roundUp(localWorkSize, ELEMENT_COUNT);
            
            out.println("With localWorkSize=" + localWorkSize + " globalWorkSize=" + globalWorkSize);

            // A, B are input buffers, C is for the result
            clBufferA = context.createFloatBuffer(ELEMENT_COUNT, READ_ONLY);
            clBufferB = context.createFloatBuffer(ELEMENT_COUNT, WRITE_ONLY);

            out.println(
                    "Allocated device memory: " +
                    (clBufferA.getCLSize()+clBufferB.getCLSize())/1000000 +
                    "MB");

            // fill input buffers with random numbers
            fillBuffer(clBufferA.getBuffer(), SEED_A);
            
            // copy data to GPU
            queue.putWriteBuffer(clBufferA, true);
            
            // call GPU
            kernel.setArg(0, clBufferA);
            kernel.setArg(1, FLOAT_TO_ADD);
            kernel.setArg(2, clBufferB);
            kernel.setArg(3, ELEMENT_COUNT);
            
            time = nanoTime();
            queue.put1DRangeKernel(kernel, 0, globalWorkSize, localWorkSize);
            queue.finish();
            time = nanoTime() - time;

            out.println("OCL AddFloat took " + (time/1000000) + "ms");
            
            // get results
            queue.putReadBuffer(clBufferB, true);
            seqBufferA = FloatBuffer.allocate(ELEMENT_COUNT);
            seqBufferB = FloatBuffer.allocate(ELEMENT_COUNT);

            fillBuffer(seqBufferA, SEED_A);
            
            time = nanoTime();
            for (int i=0; i<ELEMENT_COUNT; ++i)
                seqBufferB.put(i, seqBufferA.get(i) + FLOAT_TO_ADD);
            time = nanoTime() - time;
            
            out.println("Sequential AddFloat took " + (time/1000000) + "ms");
            
            out.print("Verifying results... ");
            passed = true;
            for (int i=0; i<ELEMENT_COUNT; ++i)
                if (seqBufferB.get(i) != clBufferB.getBuffer().get(i)) {
                    passed = false;
                    break;
                }
            
            if (passed)
                out.println("OK");
            else
                out.println("FAILED");
            
            out.println();
                        
            clBufferA.release();
            clBufferB.release();
            
            kernel.release();
            
/* Reverse */
            // create CL kernel based on the previously built program
            kernel = program.createCLKernel("Reverse");

            localWorkSize = min((int)kernel.getWorkGroupSize(device), MAX_WORK_SIZE);
            globalWorkSize = roundUp(localWorkSize, ELEMENT_COUNT);
            
            out.println("With localWorkSize=" + localWorkSize + " globalWorkSize=" + globalWorkSize);

            // A, B are input buffers, C is for the result
            clBufferA = context.createFloatBuffer(ELEMENT_COUNT, READ_ONLY);
            clBufferB = context.createFloatBuffer(ELEMENT_COUNT, WRITE_ONLY);

            out.println(
                    "Allocated device memory: " +
                    (clBufferA.getCLSize()+clBufferB.getCLSize())/1000000 +
                    "MB");

            // fill input buffers with random numbers
            fillBuffer(clBufferA.getBuffer(), SEED_A);
            
            // copy data to GPU
            queue.putWriteBuffer(clBufferA, true);
            
            // call GPU
            kernel.setArg(0, clBufferA);
            kernel.setArg(1, clBufferB);
            kernel.setArg(2, ELEMENT_COUNT);
            
            time = nanoTime();
            queue.put1DRangeKernel(kernel, 0, globalWorkSize, localWorkSize);
            queue.finish();
            time = nanoTime() - time;

            out.println("OCL Reverse took " + (time/1000000) + "ms");
            
            // get results
            queue.putReadBuffer(clBufferB, true);
            seqBufferA = FloatBuffer.allocate(ELEMENT_COUNT);
            seqBufferB = FloatBuffer.allocate(ELEMENT_COUNT);

            fillBuffer(seqBufferA, SEED_A);
            
            time = nanoTime();
            for (int i=0; i<ELEMENT_COUNT; ++i)
                seqBufferB.put(i, seqBufferA.get(ELEMENT_COUNT - 1 - i));
            time = nanoTime() - time;
            
            out.println("Sequential Reverse took " + (time/1000000) + "ms");
            
            out.print("Verifying results... ");
            passed = true;
            for (int i=0; i<ELEMENT_COUNT; ++i)
                if (seqBufferB.get(i) != clBufferB.getBuffer().get(i)) {
                    passed = false;
                    break;
                }
            
            if (passed)
                out.println("OK");
            else
                out.println("FAILED");
            
            out.println();
            
            clBufferA.release();
            clBufferB.release();
            
            kernel.release();
            
            
        } finally {
            // cleanup all resources associated with this context
            context.release();
        }

    }

    private static void fillBuffer(FloatBuffer buffer, int seed) {
        Random rnd = new Random(seed);
        while(buffer.remaining() != 0)
            buffer.put(rnd.nextFloat()*100);
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