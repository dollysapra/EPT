package nl.uva.aloha.genetic;

import java.util.ArrayList;

import io.jenetics.Chromosome;
import io.jenetics.util.ISeq;
import io.jenetics.util.IntRange;
import io.jenetics.util.MSeq;

public class SkipCnxnDualLayerChromosome extends DualLayerChromosome {

	/**
	 * 
	 */
	private static final long serialVersionUID = 8454873894457436656L;


	protected SkipCnxnDualLayerChromosome(ISeq<? extends LayerGene> genes, IntRange lengthRange) 
	{
		super(genes, lengthRange);
	}
	
	public SkipCnxnDualLayerChromosome(String layerTypes, final int minNeurons, final int maxNeurons,final IntRange layerRange) 
	{
		 this(MSeq.<LayerGene>of(getDualGeneSequenceWithSkipLayer(layerTypes,minNeurons,maxNeurons-1, layerRange)).toISeq(), layerRange);
		 //maxneuron -1 to get 
		_valid = true;
	}
	
	protected static ArrayList<LayerGene> getDualGeneSequenceWithSkipLayer(String layerTypes, final int minNeurons, final int maxNeurons, final IntRange layerRange)
	{
		if(layerRange.getMin() < 2)
			return null;
		ArrayList<LayerGene> layers = getDualGenesSequence(layerTypes,minNeurons,maxNeurons, IntRange.of(layerRange.getMin(), layerRange.getMax()-2));
	    
		LayerGene skipConvLayerGene = new LayerGene("Skip",layers.get(layers.size() -2).getLayer().getNeuronsNum()); //Last is Relu. -2 is Conv
		skipConvLayerGene.setMaxNumNeurons(maxNeurons);
		skipConvLayerGene.setMinNumNeurons(minNeurons);
		
		LayerGene aadLayerGene = new LayerGene("Add",1);
		
		
		layers.add(skipConvLayerGene);
		layers.add(aadLayerGene);
		
		return layers;
	}
	

	public static SkipCnxnDualLayerChromosome of(String layerType, final int minNeurons, final int maxNeurons,final IntRange lengthRange)
	{
		return new SkipCnxnDualLayerChromosome(layerType,  minNeurons, maxNeurons,lengthRange);
	}
	

	@Override
	public Chromosome<LayerGene> newInstance(ISeq<LayerGene> genes) 
	{
		return new SkipCnxnDualLayerChromosome(genes, lengthRange());
	}
		
	

	@Override
	public Chromosome<LayerGene> newInstance() {
		return new SkipCnxnDualLayerChromosome(_layerTypes, getMinNeurons(), getMaxNeurons(),lengthRange());
	}

}
