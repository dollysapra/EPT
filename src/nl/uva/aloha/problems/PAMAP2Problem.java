package nl.uva.aloha.problems;

import java.io.File;
import java.util.function.Function;

import espam.datamodel.graph.cnn.Network;
import espam.datamodel.graph.csdf.datasctructures.CSDFEvalResult;
import io.jenetics.engine.Codec;
import io.jenetics.engine.Problem;
import io.jenetics.ext.moea.Vec;
import nl.uva.aloha.Alterers.ONNXAlteration;
import nl.uva.aloha.helpers.Configs;
import nl.uva.aloha.helpers.OnnxHelper;
import nl.uva.aloha.helpers.OnnxRegistry;

public class PAMAP2Problem implements Problem<GenotypeTranslation, LayerGene, Vec<double[]>> {

	private static final int OBJECTIVES = 2; 
	
	@Override
	public Codec<GenotypeTranslation, LayerGene> codec() 
	{
		return new DNNCodec();
	}

	@Override
	public Function<GenotypeTranslation, Vec<double[]>> fitness()
	{
		return this::fitnessEvaluations;
	}	

	
	private Vec<double[]> fitnessEvaluations(GenotypeTranslation gtTrans)
	{
		final double[] fitnessValues = new double[2];
		
		double accuracy = accuracyFitness(gtTrans.getOnnxName());
		gtTrans.setAccuracy(accuracy);
		fitnessValues[0] = (1.0 - accuracy ); //GA engine is set to minimizing - So minimize the error. 
		
		ONNXAlteration onnxAlterGt = new ONNXAlteration(OnnxRegistry._onnxFolder + gtTrans.getOnnxName());
		fitnessValues[1] = onnxAlterGt.countParamsAssumeBN(); 
		
		System.out.println(gtTrans.getOnnxName()  + ":" + fitnessValues[0] + ":params:" + fitnessValues[1] + "(" + onnxAlterGt.countParams()+")");
		
		return Vec.of(fitnessValues);
	}
	
	
	
	private static double accuracyFitness(String onnxName) 
	{
		try
		{
		 Double accuracyResult = OnnxHelper.evaluatePamap2Onnx(onnxName, Configs.generationTrack.intValue());
		 if(accuracyResult!=null)
			{
				 return accuracyResult;
			}
			else 
			{
				System.err.println("Accuracy Result is null");
				return 0.0;
			}
		}
		catch(Exception e)
		{
			System.err.print(e.getStackTrace());
			return 0.0;
		}
			
	
	}
	
	
	private CSDFEvalResult hardwarefitness(Network network) 
	{
		File tempFile = new File("");
        String platformPath = tempFile.getAbsolutePath() + "/nl/uva/aloha/platforms/Jetson.json";
        
		if(network.checkConsistency())
		{
			try
			{
				CSDFEvalResult evalResult =  espam.main.cnnUI.UI.getInstance().evaluate(network, platformPath);

				return evalResult;  //evalResult.getPerformance(); //Double.MAX_VALUE;
				
			}
			catch(Exception e)
			{
				System.err.print(e.getStackTrace());
				return null;
			}
		}
		else
		{
			System.err.print("Error: This should never be printed ideally");
			return null;
		}
	}
	
	
}

