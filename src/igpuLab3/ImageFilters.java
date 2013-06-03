package igpuLab3;

import com.jogamp.common.nio.Buffers;
import com.jogamp.opencl.CLBuffer;
import com.jogamp.opencl.CLCommandQueue;
import com.jogamp.opencl.CLContext;
import com.jogamp.opencl.CLDevice;
import com.jogamp.opencl.CLKernel;
import com.jogamp.opencl.CLProgram;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.FloatBuffer;
import java.util.Arrays;
import javax.imageio.ImageIO;
import javax.swing.ImageIcon;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.SwingUtilities;
import org.omg.PortableInterceptor.SYSTEM_EXCEPTION;

public class ImageFilters {
    
    private final int V_OFFSET = 30;
    private final int H_OFFSET = 75;
    private final String PROGRAM_FILENAME = "ImageFilters.cl";
    private final String IMAGE_FILENAME = "lena20.jpg";

    private CLContext _context;
    private CLDevice _device;
    private CLCommandQueue _commandQueue;
    private CLProgram _program;
    private BufferedImage _image;
    private static int windowNo = 0;

    
    /* public interface */

    public ImageFilters(CLContext context, CLDevice device) throws IOException {
        _context = context;
        _device = device;
        
        System.out.println("Using device: " + _device);
        System.out.println("CL_DEVICE_MAX_WORK_GROUP_SIZE=" + _device.getMaxWorkGroupSize());
        System.out.println("CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS=" + _device.getMaxWorkItemDimensions());
        System.out.println("CL_DEVICE_MAX_WORK_ITEM_SIZES=" + Arrays.toString(_device.getMaxWorkItemSizes()));
        
        _commandQueue = _device.createCommandQueue();
        
        _program = _context.createProgram(getClass().getResourceAsStream(PROGRAM_FILENAME));
        _program.build(CLProgram.CompilerOptions.FAST_RELAXED_MATH);
        
        _image = ImageIO.read(getClass().getResourceAsStream(IMAGE_FILENAME));

        showImage(_image, 0, V_OFFSET * ++windowNo, "Original", JFrame.EXIT_ON_CLOSE);
    }
    
    public void showFiltered(String filter, float... parameters) {
        CLBuffer<FloatBuffer> buffer = _context.createBuffer(
            Buffers.newDirectFloatBuffer(
                _image.getRaster().getPixels(0, 0, _image.getWidth(), _image.getHeight(), (float[])null)
            ),
            CLBuffer.Mem.READ_WRITE);
        CLKernel kernel = _program.createCLKernel(filter);

        kernel.setArg(0, buffer);
        kernel.setArg(1, _image.getWidth());
        kernel.setArg(2, _image.getHeight());
        kernel.setArg(3, 3);

        int nextArg = 4;
        for (float p: parameters)
            kernel.setArg(nextArg++, p);
        
        int[] localWorkSize = {(int) kernel.getWorkGroupSize(_device), 1, 1};
        int[] globalWorkSize = {
            roundUp(localWorkSize[0], _image.getWidth()),
            roundUp(localWorkSize[1], _image.getHeight()),
            roundUp(localWorkSize[2], 3)
        };

        System.out.println(
                "Enqueuing kernel '" + filter +
                "' with GWS=" + Arrays.toString(globalWorkSize) +
                " and LWS=" + Arrays.toString(localWorkSize)
                );
        
        _commandQueue.putWriteBuffer(buffer, false);
        _commandQueue.put3DRangeKernel(
                kernel,
                0, 0, 0,
                globalWorkSize[0], globalWorkSize[1], globalWorkSize[2],
                localWorkSize[0], localWorkSize[1], localWorkSize[2]
                );
        _commandQueue.putReadBuffer(buffer, true);
        
        showImage(createImage(_image.getWidth(), _image.getHeight(), buffer), H_OFFSET * windowNo, V_OFFSET * ++windowNo, filter + " " + Arrays.toString(parameters));            

        kernel.release();
        buffer.release();
    }
    
    
    
    /* private methods */
    
    private BufferedImage createImage(int width, int height, CLBuffer<FloatBuffer> buffer) {
        BufferedImage image = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        float[] pixels = new float[buffer.getBuffer().capacity()];
        buffer.getBuffer().get(pixels).rewind();
        image.getRaster().setPixels(0, 0, width, height, pixels);
        return image;
    }
    
    private void showImage(BufferedImage image, int x, int y, String title) {
        showImage(image, x, y, title, JFrame.HIDE_ON_CLOSE);
    }
    
    private void showImage(final BufferedImage image, final int x, final int y, final String title, final int closeOperation) {
        SwingUtilities.invokeLater(new Runnable() {
            @Override
            public void run() {
                JFrame frame = new JFrame("ImageFilters :: " + title);
                frame.setDefaultCloseOperation(closeOperation);
                frame.add(new JLabel(new ImageIcon(image)));
                frame.pack();
                frame.setLocation(x, y);
                frame.setVisible(true);
            }
        });
    }
    
    private int roundUp(int groupSize, int globalSize) {
        int r = globalSize % groupSize;
        if (r == 0) {
            return globalSize;
        } else {
            return globalSize + groupSize - r;
        }
    }
}
