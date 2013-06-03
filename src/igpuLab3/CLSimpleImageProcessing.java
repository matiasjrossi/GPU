package igpuLab3;

import com.jogamp.opencl.CLContext;
import com.jogamp.opencl.CLDevice;
import java.io.IOException;


public class CLSimpleImageProcessing {
    
    public static void main(String[] args) throws IOException 
    {
        CLContext context = CLContext.create();
   
        try
        {
            CLDevice device = context.getMaxFlopsDevice();
            
            ImageFilters imFilters = new ImageFilters(context, device);
            
            imFilters.showFiltered("Negate");
            
            imFilters.showFiltered("Brightness", 100.0f);
            imFilters.showFiltered("Brightness", -100.0f);
            
            imFilters.showFiltered("Threshold", 100.0f);
            imFilters.showFiltered("Threshold", 200.0f);
            
            imFilters.showFiltered("Contrast", -50.0f, -45.0f);
            imFilters.showFiltered("Contrast", 0.0f, 0.5f);
            
            imFilters.showFiltered("Smoothen");
                   
        } finally {
            context.release();
        }        
    }
}