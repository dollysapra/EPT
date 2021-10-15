package nl.uva.aloha.helpers;

import java.io.BufferedReader;
import java.io.File;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Date;

import espam.datamodel.graph.cnn.Network;
import espam.utils.fileworker.ONNXFileWorker;
import io.jenetics.Genotype;
import nl.uva.aloha.Alterers.ONNXAlteration;
import nl.uva.aloha.converters.GeneToOnnx;
import nl.uva.aloha.genetic.LayerGene;
import onnx.ONNX.ModelProto;

public class OnnxHelper 
{
	
	private static String _pythonLocalFilePath = "/src/nl/uva/aloha/pythonScripts/";
	
	private static long idCounter = 1;
	
	private static int genDerived = 0;
	
	private static int vocTrainReqNum =0;

    static ArrayList<String> que = new ArrayList<String>(Arrays.asList("0","1","2","3","4","5","6","7"));


    
	static public String saveGenotypeAsOnnx(Genotype<LayerGene> gt)
	{
		ModelProto model =  new GeneToOnnx(gt).convertToONNXModel();
		return saveOnnx(model);
	}
	
	public static synchronized String getNextGPU() {
	     return que.remove(0);
	 }
	
   

	static public String saveNetworkAsOnnx(Network network)
	{
		ModelProto model =  new GeneToOnnx(network).convertToONNXModel();
		return saveOnnx(model);
	}
	
	static public String saveOnnx(ModelProto model)
	{
		String onnxname = "o" + idCounter++ + ".onnx";
		File file = new File(OnnxRegistry._onnxFolder);
		if (!file.exists()) 
		{
            System.out.print("No Folder:" + OnnxRegistry._onnxFolder);
            file.mkdir();
            System.out.println("Folder created" + new Date().toString());
        }
		
		ONNXFileWorker.writeModel(model, OnnxRegistry._onnxFolder + onnxname);		
		OnnxGarbageCollector.getInstance().addFileToListForGeneration(genDerived, onnxname);
        return onnxname;
	}
	
	static public Double evaluateOnnx(String onnxname)
	{
		
		return evaluateOnnx(onnxname, 1);
		
	}
	
	
	
	static public Double testOnnx(String onnxname)
	{
		ONNXAlteration onnxAlterGt = new ONNXAlteration(OnnxRegistry._onnxFolder + onnxname);
		onnxAlterGt.resetReshapeLayerDimForTesting();
		//onnxAlterGt.renameInputOutputLayerAfterOneRun();
		onnxAlterGt.updateONNXFile();
		
		File tempFile = new File("");
        String pythonScriptPath = tempFile.getAbsolutePath() + _pythonLocalFilePath + "Cifar10_onnx_test.py";
        if(Configs.DATASET.equals("PAMAP2"))
        	pythonScriptPath = tempFile.getAbsolutePath() + _pythonLocalFilePath + "test_pamap2.py";
        
        else if(Configs.DATASET.equals("VOC"))
        	pythonScriptPath = tempFile.getAbsolutePath() + _pythonLocalFilePath + "test_VOC.py";
 
        
        String[] cmd = new String[3];
        cmd[0] = Configs.PythonPath;
        cmd[1] = pythonScriptPath;
        cmd[2] = OnnxRegistry._onnxFolder + onnxname;
       // cmd[3] = 
        
        try
        {
        	 Runtime rt = Runtime.getRuntime();
             Process pr = rt.exec(cmd);

             /** retrieve output from python script */
             BufferedReader bfr = new BufferedReader(new InputStreamReader(pr.getInputStream()));
             String line = "";
             String pythonScriptResult = "";
             while((line = bfr.readLine()) != null) {
                 pythonScriptResult+=line;
             }
             System.out.println("Test - " +pythonScriptResult);
             return new Double(pythonScriptResult.split(":")[1]);
        }
        catch(Exception e)
        {
        	System.err.println(e.getMessage());
        	return 0.0;
        }
	}
	
	static public Double evaluatePamap2Onnx(String onnxname, Integer generation)
	{
		genDerived = generation;
		File tempFile = new File("");
        String pythonScriptPath = tempFile.getAbsolutePath() + _pythonLocalFilePath + "Pamap2_training_cpu.py";
        
        String[] cmd = new String[4];
        cmd[0] = Configs.PythonPath;
        cmd[1] = pythonScriptPath;
        cmd[2] = OnnxRegistry._onnxFolder + onnxname;
        cmd[3] = generation.toString();
        
        try
        {
        	 Runtime rt = Runtime.getRuntime();
             Process pr = rt.exec(cmd);
             int exitVal = pr.waitFor();
             //System.out.print("Exit:" +  exitVal);
             
             /** retrieve output from python script */
             BufferedReader bfr = new BufferedReader(new InputStreamReader(pr.getInputStream()));
             String line = "";
             String pythonScriptResult = "";
             while((line = bfr.readLine()) != null) {
                 pythonScriptResult+=line;
             }
             if(exitVal != 0)
             {
	             BufferedReader bfr1 = new BufferedReader(new InputStreamReader(pr.getErrorStream()));
	             line = "";
	             String pythonScriptError= "";
	             while((line = bfr1.readLine()) != null) {
	            	 pythonScriptError+=(line+"/n");
	             }
	             System.out.println("Error - " +pythonScriptError);
	             return 0.0;
             }
             System.out.print("R-" +pythonScriptResult + " -- ");
           
             if(pythonScriptResult.split(":").length>1)
            	 return new Double(pythonScriptResult.split(":")[1]);
             else 
            	 return evaluatePamap2Onnx(onnxname,generation);
        }
        catch(Exception e)
        {
        	System.err.println(e.getMessage());
        	return 0.0;
        }
	}
	
	static public Double evaluateVOCOnnx1gpu(String onnxname, Integer generation)
	{
		genDerived = generation;
		File tempFile = new File("");
        String pythonScriptPath = tempFile.getAbsolutePath() + _pythonLocalFilePath + "VOC_training.py";
        
        String[] cmd = new String[4];
        cmd[0] = Configs.PythonPath;
        cmd[1] = pythonScriptPath;
        cmd[2] = OnnxRegistry._onnxFolder + onnxname;
        cmd[3] = generation.toString();
        
        try
        {
        	 Runtime rt = Runtime.getRuntime();
             Process pr = rt.exec(cmd);
             int exitVal = pr.waitFor();
             //System.out.print("Exit:" +  exitVal);
             
             /** retrieve output from python script */
             BufferedReader bfr = new BufferedReader(new InputStreamReader(pr.getInputStream()));
             String line = "";
             String pythonScriptResult = "";
             while((line = bfr.readLine()) != null) {
                 pythonScriptResult+=line;
             }
             if(exitVal != 0)
             {
	             BufferedReader bfr1 = new BufferedReader(new InputStreamReader(pr.getErrorStream()));
	             line = "";
	             String pythonScriptError= "";
	             while((line = bfr1.readLine()) != null) {
	            	 pythonScriptError+=(line+"/n");
	             }
	             System.out.println("Error - " +pythonScriptError);
	             return 0.0;
             }
             System.out.print("R-" +pythonScriptResult + " -- ");
           
             if(pythonScriptResult.split("-").length>1)
             {
            	 String[] splits = pythonScriptResult.split(":");
            	 
            	 
            	 return new Double(splits[splits.length-1]);
             }
             else 
            	 return evaluateVOCOnnx(onnxname,generation);
        }
        catch(Exception e)
        {
        	System.err.println(e.getMessage());
        	return 0.0;
        }
	}
	
	
	static public Double evaluateVOCOnnx(String onnxname, Integer generation)
    {

            return evaluatemultiGPUKeras("VOC_training.py",onnxname, generation);
    }

    static public Double evaluateCifarOnnxKeras(String onnxname, Integer generation)
    {
    			   
            return evaluatemultiGPUKeras("CIFAR_training_keras.py",onnxname, generation);
    }

    static public Double evaluatemultiGPUKeras(String scriptname,String onnxname, Integer generation)
    {
            genDerived = generation;
            File tempFile = new File("");
            //String gpu = String.valueOf(vocTrainReqNum%4);

            //System.out.print("Queue:" +  que.toString());

            String gpu =getNextGPU();
            System.out.print("gput:" +  gpu);
            vocTrainReqNum = vocTrainReqNum+1;
            String pythonScriptPath = tempFile.getAbsolutePath() + _pythonLocalFilePath + scriptname;

            String[] cmd = new String[5];
            cmd[0] = Configs.PythonPath;
            cmd[1] = pythonScriptPath;
            cmd[2] = OnnxRegistry._onnxFolder + onnxname;
            cmd[3] = generation.toString();
            cmd[4] = gpu;

            try
            {
            Runtime rt = Runtime.getRuntime();
            Process pr = rt.exec(cmd);
            int exitVal = pr.waitFor();

            /** retrieve output from python script */
            BufferedReader bfr = new BufferedReader(new InputStreamReader(pr.getInputStream()));
            String line = "";
            String pythonScriptResult = "";
            while((line = bfr.readLine()) != null) {
                    pythonScriptResult+=line;
            }
            if(exitVal != 0)
            {
                 BufferedReader bfr1 = new BufferedReader(new InputStreamReader(pr.getErrorStream()));
                 line = "";
                 String pythonScriptError= "";
                 while((line = bfr1.readLine()) != null) {
                     pythonScriptError+=(line+"/n");
                 }
                 System.out.println("Error - " +pythonScriptError);
                 return 0.0;
            }
            System.out.print("R-" +pythonScriptResult + " -- ");

            if(pythonScriptResult.split("-").length>1)
            {
                     String[] splits = pythonScriptResult.split(":");


                    return new Double(splits[splits.length-1]);
            }

            else
                return evaluatemultiGPUKeras(scriptname, onnxname,generation);
        }
         catch(Exception e)
        {
                System.out.print("cmd:error"+ cmd.toString());
                System.err.println(e.getMessage());
                return 0.0;
        }
        finally
         {
                que.add(gpu);
        }
}

	
	
	static public Double evaluateOnnxMulti(String onnxname)
	{
		return evaluateOnnx(onnxname,1);
	}
	static public Double evaluateOnnx(String onnxname, Integer generation)
	{
		genDerived = generation;
		File tempFile = new File("");
        String pythonScriptPath = tempFile.getAbsolutePath() + _pythonLocalFilePath + "Cifar10_onnx_shuffle_gen.py";
        
        String[] cmd = new String[4];
        cmd[0] = Configs.PythonPath;
        cmd[1] = pythonScriptPath;
        cmd[2] = OnnxRegistry._onnxFolder + onnxname;
        cmd[3] = generation.toString();
        
        try
        {
        	 Runtime rt = Runtime.getRuntime();
             Process pr = rt.exec(cmd);
             int exitVal = pr.waitFor();
             //System.out.print("Exit:" +  exitVal);
             
             /** retrieve output from python script */
             BufferedReader bfr = new BufferedReader(new InputStreamReader(pr.getInputStream()));
             String line = "";
             String pythonScriptResult = "";
             while((line = bfr.readLine()) != null) {
                 pythonScriptResult+=line;
             }
             if(exitVal !=0)
             {
	             BufferedReader bfr1 = new BufferedReader(new InputStreamReader(pr.getErrorStream()));
	             line = "";
	             String pythonScriptError= "";
	             while((line = bfr1.readLine()) != null) {
	            	 pythonScriptError+=(line+"/n");
	             }
	             System.out.println("Error - " +pythonScriptError);
	             return 0.0;
             }
             System.out.print("R-" +pythonScriptResult + "...");
           
           
             if(pythonScriptResult.split(":").length>1)
            	 return new Double(pythonScriptResult.split(":")[1]);
             else 
            	 return evaluateOnnx(onnxname,generation);
        }
        catch(Exception e)
        {
        	System.err.println(e.getMessage());
        	return 0.0;
        }
	}
	
}

