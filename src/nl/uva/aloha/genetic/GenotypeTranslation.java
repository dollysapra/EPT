package nl.uva.aloha.genetic;

import java.util.Iterator;

import espam.datamodel.graph.cnn.Network;
import espam.datamodel.graph.csdf.datasctructures.CSDFEvalResult;
import io.jenetics.AbstractChromosome;
import io.jenetics.Chromosome;
import io.jenetics.Genotype;
import nl.uva.aloha.converters.GeneToNetwork;
import nl.uva.aloha.helpers.OnnxHelper;
import nl.uva.aloha.helpers.OnnxRegistry;

public class GenotypeTranslation 
{

	private Genotype<LayerGene> _gt;
	private Boolean _valid;
	private Network _network;
	private String _onnxName;
	
	private double _accuracy;
	private CSDFEvalResult _hardwareEval;
	
	public GenotypeTranslation(Genotype<LayerGene> genotype)
	{
		
		
		_valid = true;
		
		_gt = genotype;
		
		if(!((Genotype<LayerGene>) _gt ).isValid())
			_valid = false;
		
		_network = GeneToNetwork.createNetworkFromGenotype((Genotype<LayerGene>) _gt);
		if(_network == null)
			_valid =  false;
		
		if(!_network.checkConsistency())
			_valid =  false;
		
		int hashcode = System.identityHashCode(_gt);
		
		if(OnnxRegistry.getInstance().hasEntry(hashcode))
		{
			_onnxName = OnnxRegistry.getInstance().getEntry(hashcode);
				return;
		}		
		_onnxName = OnnxHelper.saveNetworkAsOnnx(_network);
		OnnxRegistry.getInstance().addEntry(hashcode, _onnxName);
		//System.out.println(_gt.hashCode() + ":" + _onnxName);
		
	}
	
	public static String getSignature(Genotype<LayerGene> gt)
	{
		String signature = "";
		
		for (Iterator<Chromosome<LayerGene>> i = gt.iterator(); i.hasNext(); ) 
		{
			AbstractChromosome<LayerGene> ac = (AbstractChromosome<LayerGene>)(i.next());
			for (Iterator<LayerGene> j = ac.iterator(); j.hasNext(); ) 
					signature = signature + j.next()._layerType.charAt(0);
			
		}
		
		return signature;
	}
	
	//TODO: THIS FUNCTION MAYBE NOT NEEDED 
	public static GenotypeTranslation decodeGenotype(Genotype<LayerGene> genotype)
	{
		return new GenotypeTranslation(genotype);
	}
	
	public Boolean isValid()
	{
		return _valid;
	}
	
	public Network getNetwork()
	{
		return _network;
	}
	
	public String getOnnxName()
	{
		return _onnxName;
	}
	
	
	@Override
	public String toString() 
	{
		return _onnxName + "-----" + _network.toString();
	}
	
	public String toEvaluatedString()
	{
		if(_hardwareEval == null)
			return "Not evaluated yet";
		return _onnxName + ":" + _accuracy + ":" + _hardwareEval.toString();
	}
	public void setAccuracy(double accuracy)
	{
		_accuracy = accuracy;
	}
	
	public void setHardwareEval(CSDFEvalResult evalResult)
	{
		_hardwareEval = evalResult;
	}
}
