package nl.uva.aloha.problems;

import java.util.function.Function;

import espam.datamodel.graph.cnn.Network;
import espam.datamodel.graph.csdf.datasctructures.CSDFEvalResult;
import espam.interfaces.python.Espam2DARTS;
import io.jenetics.engine.Codec;
import io.jenetics.engine.Problem;
import io.jenetics.ext.moea.Vec;
import nl.uva.aloha.helpers.OnnxHelper;

public class AlohaProblem implements Problem<GenotypeTranslation, LayerGene, Vec<double[]>> {

	private static final int OBJECTIVES = 3; //1-Accuracy //2-Memory //3-Energy
	
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
		
		double accuracy = accuracyFitness(gtTrans.getOnnxName());
		gtTrans.setAccuracy(accuracy);
		fitnessValues[0] = (1.0 - accuracy ); //GA engine is set to minimizing - So minimize the error. 
		
		CSDFEvalResult evalRes = hardwarefitness(gtTrans.getNetwork());
		gtTrans.setHardwareEval(evalRes);
		if(evalRes==null)
		{
			fitnessValues[1] = Double.MAX_VALUE;
			fitnessValues[2] = Double.MAX_VALUE;
		}
		else
		{
			fitnessValues[1] = evalRes.getMemory();
			fitnessValues[2] = evalRes.getPerformance();
		}
		
		System.out.println(gtTrans.toEvaluatedString());
		
		return Vec.of(fitnessValues);
	}
	
	
	private static double accuracyFitness(String onnxName) 
	{
		try
		{
		 Double accuracyResult = OnnxHelper.evaluateOnnxMulti(onnxName);
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
		if(network.checkConsistency())
		{
			try
			{
				CSDFEvalResult evalResult =  new Espam2DARTS().evaluateCNN(network);
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
