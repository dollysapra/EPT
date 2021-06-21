package nl.uva.aloha.genetic;


import static io.jenetics.util.RandomRegistry.getRandom;

import io.jenetics.Chromosome;
import io.jenetics.VariableChromosome;
import io.jenetics.util.ISeq;
import io.jenetics.util.IntRange;
import io.jenetics.util.MSeq;
import io.jenetics.util.Verifiable;

public class SimpleLayerChromosome
		extends VariableChromosome<LayerGene>
	
{

	private static final long serialVersionUID = 1312894805838650579L;

	//Constraints of each gene
	private final int _minNumNeurons;
	private final int _maxNumNeurons;
	private final String _layerType;
	
	
	protected SimpleLayerChromosome(ISeq<? extends LayerGene> genes, IntRange lengthRange) 
	{
		super(genes,lengthRange);
		_minNumNeurons = genes.get(0).getMinNumNeurons();
		_maxNumNeurons = genes.get(0).getMaxNumNeurons();
		_layerType = genes.get(0).getLayerType();
	}
	
	
	public SimpleLayerChromosome(String layerType, final int minNeurons, final int maxNeurons,final IntRange layerRange) 
	{
		 this((MSeq.<LayerGene>ofLength(getRandom().nextInt(layerRange.size())+layerRange.getMin())
					.fill(() -> new LayerGene(layerType, minNeurons, maxNeurons)))
					.toISeq(), layerRange);
		 
		_valid = true;
	}
	

	/**
	 * Create a new  chromosome of length one.
	 */
	public SimpleLayerChromosome(String layerType, final int minNeurons, final int maxNeurons) 
	{
		this(layerType,minNeurons, maxNeurons, IntRange.of(1));
	}
	
	

	
	
	
	public static SimpleLayerChromosome of(String layerType, final int minNeurons, final int maxNeurons)
	{
		return new SimpleLayerChromosome(layerType,  minNeurons, maxNeurons);
	}
	
	public static SimpleLayerChromosome of(String layerType, final int minNeurons, final int maxNeurons,final IntRange lengthRange)
	{
		return new SimpleLayerChromosome(layerType,  minNeurons, maxNeurons,lengthRange);
	}
	
	@Override
	public boolean isValid() 
	{		
		if (_valid == null) 
		{
			_valid = _genes.forAll(Verifiable::isValid);
		}

	return _valid;
	}
	
	@Override
	public Chromosome<LayerGene> newInstance(ISeq<LayerGene> genes) 
	{
		return new SimpleLayerChromosome(genes, lengthRange());
	}

	@Override
	public Chromosome<LayerGene> newInstance() 
	{
		return new SimpleLayerChromosome(_layerType, _minNumNeurons, _maxNumNeurons,lengthRange());
	}

	
	public int getMinNeurons()
	{
		return _minNumNeurons;
	}
	
	public int getMaxNeurons()
	{
		return _maxNumNeurons;
	}
	
	public String getLayerType()
	{
		return _layerType;
	}
	

}
