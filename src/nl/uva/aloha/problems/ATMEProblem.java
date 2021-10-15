package nl.uva.aloha.problems;

import java.io.File;
import java.util.function.Function;

import espam.datamodel.graph.cnn.Network;
import espam.datamodel.graph.csdf.datasctructures.CSDFEvalResult;
import espam.parser.onnx.ONNX2CNNConverter;
import espam.utils.fileworker.ONNXFileWorker;
import io.jenetics.engine.Codec;
import io.jenetics.engine.Problem;
import io.jenetics.ext.moea.Vec;
import nl.uva.aloha.helpers.Configs;
import nl.uva.aloha.helpers.OnnxHelper;
import nl.uva.aloha.helpers.OnnxRegistry;
import onnx.ONNX.ModelProto;

public class ATMEProblem implements Problem<GenotypeTranslation, LayerGene, Vec<double[]>> {

	private static final int OBJECTIVES = 4; //1-Accuracy //2-Memory //3-Energy //4-Performance
	
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
		final double[] fitnessValues = new double[OBJECTIVES];
		
		CSDFEvalResult evalRes = hardwarefitness(gtTrans.getOnnxName());
		double accuracy = accuracyFitness(gtTrans.getOnnxName());
		gtTrans.setAccuracy(accuracy);
		fitnessValues[0] = (1.0 - accuracy ); //GA engine is set to minimizing - So minimize the error. 
		
		
				
        		
	
		
		gtTrans.setHardwareEval(evalRes);
		if(evalRes==null)
		{
			fitnessValues[1] = Double.MAX_VALUE;
			fitnessValues[2] = Double.MAX_VALUE;
			fitnessValues[3] = Double.MAX_VALUE;
		}
		else
		{
			fitnessValues[1] = evalRes.getMemory();
			fitnessValues[2] = evalRes.getEnergy();
			fitnessValues[3] = evalRes.getPerformance();
		}
		
		System.out.println(gtTrans.toEvaluatedString());
		
		return Vec.of(fitnessValues);
	}
	
	
	private static double accuracyFitness(String onnxName) 
	{
		
		//return Math.random();
		try
		{
			
			Double accuracyResult = null;
			
			if(Configs.DATASET == Configs.VOCDATASET) {
				accuracyResult = OnnxHelper.evaluateVOCOnnx(onnxName, Configs.generationTrack.intValue());
			}
			else if(Configs.DATASET == Configs.PAMAPDATASET){
				 accuracyResult = OnnxHelper.evaluatePamap2Onnx(onnxName, Configs.generationTrack.intValue());
			}
			else{
				accuracyResult = OnnxHelper.evaluateCifarOnnxKeras(onnxName, Configs.generationTrack.intValue());
			}
			
			
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
	
	
	private CSDFEvalResult hardwarefitness(String onnxname) 
	{
		File tempFile = new File("");
        String platformPath = tempFile.getAbsolutePath() + "/src/nl/uva/aloha/platforms/Jetson.json";
        ModelProto model = ONNXFileWorker.readModel(OnnxRegistry._onnxFolder + onnxname);
		Network network =  ONNX2CNNConverter.convertModel(model, platformPath);
		
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
