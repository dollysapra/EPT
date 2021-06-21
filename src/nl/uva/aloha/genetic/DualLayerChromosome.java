package nl.uva.aloha.genetic;

import static io.jenetics.util.RandomRegistry.getRandom;

import java.util.ArrayList;

import io.jenetics.Chromosome;
import io.jenetics.util.ISeq;
import io.jenetics.util.IntRange;
import io.jenetics.util.MSeq;
import io.jenetics.util.RandomRegistry;

public class DualLayerChromosome 
		extends SimpleLayerChromosome //Base Gene type is still LayerGene! Dual Layer Gene just adds 2 layerGenes to the Chromosome together. 
{
	
	private static final long serialVersionUID = -47539692309150227L;
	protected final String _layerTypes;
	
	protected DualLayerChromosome(ISeq<? extends LayerGene> genes, IntRange lengthRange) 
	{
		//super(genes,lengthRange);
		super(genes,lengthRange);
		_layerTypes = genes.get(0).getLayerType() + ":" + genes.get(1).getLayerType(); //dualChromosome's first two layers reflect the layers of this instance
	}

	
	public DualLayerChromosome(String layerTypes, final int minNeurons, final int maxNeurons,final IntRange layerRange) 
	{
		 this(MSeq.<LayerGene>of(getDualGenesSequence(layerTypes,minNeurons,maxNeurons, layerRange)).toISeq(), layerRange);
		 
		_valid = true;
	}
	
	protected static ArrayList<LayerGene> getDualGenesSequence(String layerTypes, final int minNeurons, final int maxNeurons, final IntRange layerRange)
	{
		
		if(layerRange.getMin() < 2)
			return null;
		
		int maxEvenSize = layerRange.getMax() + (layerRange.getMax()%2); //Max size is exclusive for range
		int minEvenSize = layerRange.getMin() + (layerRange.getMin()%2); //Min size is inclusive for range
				
		int numDualLayers = getRandom().nextInt(maxEvenSize - minEvenSize) + minEvenSize;
		numDualLayers = numDualLayers/2;// - numDualLayers%2;
		
		ArrayList<LayerGene> layers = new ArrayList<LayerGene>(); 
		String firstLayerType = layerTypes.split(":")[0];
		String secondLAyerType = layerTypes.split(":")[1];
		
		for(int i=0; i<numDualLayers; i++)
		{
			int numberofNeurons = LayerGene.getRandomNumNeuron(minNeurons, maxNeurons); 
			
			LayerGene firstLayerGene = new LayerGene(firstLayerType,numberofNeurons);
			firstLayerGene.setMaxNumNeurons(maxNeurons);
			firstLayerGene.setMinNumNeurons(minNeurons);
			
			LayerGene secondLayerGene = new LayerGene(secondLAyerType,numberofNeurons);
			
			
			layers.add(firstLayerGene);
			layers.add(secondLayerGene);	
		}
		return layers;
	}
	
	
	public static DualLayerChromosome of(String layerType, final int minNeurons, final int maxNeurons)
	{
		return new DualLayerChromosome(layerType,  minNeurons, maxNeurons,IntRange.of(2));
	}
	
	public static DualLayerChromosome of(String layerType, final int minNeurons, final int maxNeurons,final IntRange lengthRange)
	{
		return new DualLayerChromosome(layerType,  minNeurons, maxNeurons,lengthRange);
	}

	
	
	@Override
	public Chromosome<LayerGene> newInstance(ISeq<LayerGene> genes) 
	{
		return new DualLayerChromosome(genes, lengthRange());
	}
		
	

	@Override
	public Chromosome<LayerGene> newInstance() {
		return new DualLayerChromosome(_layerTypes, getMinNeurons(), getMaxNeurons(),lengthRange());
	}

}
