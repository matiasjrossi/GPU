package igpuLab0;

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
import java.nio.ByteOrder;

/**
 * Hello Java OpenCL example. Adds all elements of buffer A to buffer B
 * and stores the result in buffer C.<br/>
 * Sample was inspired by the Nvidia VectorAdd example written in C/C++
 * which is bundled in the Nvidia OpenCL SDK.
 * @author Michael Bien
 */
public class Lab0 {

    public static void main(String[] args) throws IOException 
    {

        // set up (uses default CLPlatform and creates context for all devices)
        CLContext context = CLContext.create();
        out.println("created "+context);
        
        // always make sure to release the context under all circumstances
        // not needed for this particular sample but recommented
        try{
            // select fastest device
            CLDevice device = context.getMaxFlopsDevice();
            out.println("detected devices:" + context.getDevices().length);
            
            out.println("using "+device);

            // create command queue on device.
            CLCommandQueue queue = device.createCommandQueue();

            int elementCount = 14444771;                                  // Length of arrays to process

            // load sources, create and build program
            CLProgram program = context.createProgram(Lab0.class.getResourceAsStream("Lab0.cl"));
            program.build();

            // get a reference to the kernel function with the name 'VectorAdd'
            // and map the buffers to its input parameters.
            CLKernel kernel = program.createCLKernel("FillVector");
            
            int localWorkSize = min((int)kernel.getWorkGroupSize(device), 512);  // Local work size dimensions
            int globalWorkSize = roundUp(localWorkSize, elementCount);   // rounded up to the nearest multiple of the localWorkSize
            out.println("localWorkSize: " + localWorkSize + "; globalWorkSize: " + globalWorkSize);
            
            // A, B are input buffers, C is for the result
            CLBuffer<FloatBuffer> clBufferA = context.createFloatBuffer(elementCount/*globalWorkSize*/, READ_WRITE);
            
            out.println("used device memory: "+ (clBufferA.getCLSize())/1000000 +"MB");

            // fill input buffers with random numbers
            // (just to have test data; seed is fixed -> results will not change between runs).
            fillBuffer(clBufferA.getBuffer(), 12345);
            
           
            kernel.setArg(0, clBufferA);            
            kernel.setArg(1, elementCount);
            // followed by blocking read to get the computed results back.

            
            // Copy data to GPU
            queue.putWriteBuffer(clBufferA, false);            
            long time = nanoTime();
            // Call GPU
            queue.put1DRangeKernel(kernel, 0, globalWorkSize, localWorkSize);
                        
            // blocking read  
            queue.putReadBuffer(clBufferA, true);
            time = nanoTime() - time;

            // print first few elements of the resulting buffer to the console.
            out.println("a+b=c results snapshot: ");
             for(int i = 0; i < 10; i++)
            {
                out.print(clBufferA.getBuffer().get() + ", ");
            }
            out.println();

            out.println("gpu computation took: "+(time/1000000)+"ms");
            
            out.println("running sequential code...");
            FloatBuffer seqBufferA = FloatBuffer.allocate(elementCount);
            fillBuffer(seqBufferA, 12345);
            
            time = nanoTime();
            for (int i=0; i<elementCount; ++i)
                seqBufferA.put(i, seqBufferA.get(i) + 1);
            time = nanoTime() - time;
            out.println("secuential computation took: "+(time/1000000)+"ms");
            
            out.print("verifying... ");
            boolean correct = true;
            for (int i=0; i<elementCount; ++i)
                if (seqBufferA.get(i) != clBufferA.getBuffer().get(i)) {
                    correct = false;
                    break;
                }
            
            if (correct == true)
                out.println("OK");
            else
                out.println("FAILED");
            
        }
        finally
        {
            // cleanup all resources associated with this context.
            context.release();
        }

    }
   // Method for initializing a Buffer
    private static void fillBuffer(FloatBuffer buffer, int seed) {
        Random rnd = new Random(seed);
        while(buffer.remaining() != 0)
            buffer.put(rnd.nextFloat()*100);
        buffer.rewind();
    }
// roundUp value tu neares int
    private static int roundUp(int groupSize, int globalSize) {
        int r = globalSize % groupSize;
        if (r == 0) {
            return globalSize;
        } else {
            return globalSize + groupSize - r;
        }
    }

}