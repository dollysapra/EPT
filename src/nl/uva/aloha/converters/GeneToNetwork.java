package nl.uva.aloha.converters;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

import espam.datamodel.graph.cnn.Layer;
import espam.datamodel.graph.cnn.Network;
import espam.datamodel.graph.cnn.neurons.arithmetic.Arithmetic;
import espam.datamodel.graph.cnn.neurons.cnn.Convolution;
import espam.datamodel.graph.cnn.neurons.cnn.Pooling;
import espam.datamodel.graph.cnn.neurons.neurontypes.ArithmeticOpType;
import espam.datamodel.graph.cnn.neurons.simple.DenseBlock;
import espam.datamodel.graph.csdf.datasctructures.Tensor;
import io.jenetics.AbstractChromosome;
import io.jenetics.Chromosome;
import io.jenetics.Genotype;
import nl.uva.aloha.genetic.LayerGene;
import nl.uva.aloha.helpers.Configs;

public class GeneToNetwork 
{
	 final static Tensor INPUT_DATA_CIFAR_SHAPE = new Tensor(32,32,3);
	 final static Tensor INPUT_PAMAP2_SLIDING_SHAPE = new Tensor(100,1,40);
	 final static Tensor INPUT_TINY_YOLO_SHAPE = new Tensor(384,384,3);
	
	//DOES NOT SPECIFY CONNECTION TYPE 
	//TODO: Add connectionParameters
	private static Network createNetworkFromLayerGenes(List<?> layers)
	{
		Network network = new Network();
		Tensor inputDataShapeExample = INPUT_DATA_CIFAR_SHAPE;
		if(Configs.DATASET.equals("PAMAP2"))
			 inputDataShapeExample = INPUT_PAMAP2_SLIDING_SHAPE;
		if(Configs.DATASET.equals("TINYYOLO"))
			 inputDataShapeExample = INPUT_TINY_YOLO_SHAPE;
		if(Configs.DATASET.equals("VOC"))
			 inputDataShapeExample = INPUT_TINY_YOLO_SHAPE;
		
		
		int i =0;
		try
		{
				if(layers.get(0) instanceof LayerGene )
				{
					Network.addInputLayer(network, inputDataShapeExample, "dataI");
					//("input", ((LayerGene)layers.get(0)).getAllele().getNeuron(),1);
				}
				
		}
		catch(Exception e) {
			System.err.println(e.getMessage());
			return null;
		}
	
		Layer firstChromosomeInput = null;
		Layer lastskipConv = null;
		
		for(i=1;i<layers.size();i++) 
		{
			if(layers.get(i) instanceof LayerGene )
			{
				Layer la = ((LayerGene)layers.get(i)).getAllele();
				if(la.getName().startsWith("f1"))
				{
					firstChromosomeInput = network.getLastLayer();
					la.setName(la.getName().substring(2));
				}
				try 
				{
					if( (la.getNeuron() instanceof DenseBlock)  && (i<layers.size()-1) )
					{
						Layer nextLayer = ((LayerGene)layers.get(i+1)).getAllele();
						if((nextLayer.getName().contains("softmax")))
						{	
							DenseBlock db = (DenseBlock)la.getNeuron();
							db.setNeuronsNum(nextLayer.getNeuronsNum());
						}
						network.stackLayer(la.getName(),la.getNeuron(),la.getNeuronsNum(),la.getPads());
					}
					else if(la.getNeuron() instanceof Pooling)
					{
						network.stackLayer(la.getName(),la.getNeuron(),la.getNeuronsNum(),la.getPads());
					}
					else if ((la.getNeuron() instanceof Convolution) && ( la.getName().startsWith("skip")))
					{
						lastskipConv = la;
						network.addLayer(la.getName(),la.getNeuron(),la.getNeuronsNum(),la.getPads());
						if(firstChromosomeInput!=null)
						{
						
							network.addConnection(firstChromosomeInput.getName(), la.getName());
						}
					}
					else if(la.getNeuron() instanceof Arithmetic)
					{
						Layer prevLayer = null;
						if(i>2)
							prevLayer = ((LayerGene)layers.get(i-2)).getAllele();
					
						
						/*int dimSize = la.getNeuron().getInputDataFormat().getDimensionality();
						Neuron neuron = la.getNeuron();
						Tensor dataFormat = neuron.getInputDataFormat();
						dataFormat.setDimSize(dimSize-1, prevLayer.getNeuronsNum());*/
						
						
						Arithmetic add = new Arithmetic(ArithmeticOpType.ADD);
						Layer tempL = new Layer(la.getName(), add,1);
						
						network.addLayer(tempL.getName(),tempL.getNeuron(),1,la.getPads());
		
						
						
						if(lastskipConv!=null)
							network.addConnection(lastskipConv.getName(), la.getName());
						if(prevLayer != null)
							network.addConnection(prevLayer.getName(), tempL.getName());
					}
					else
						network.stackLayer(la.getName(),la.getNeuron(),la.getNeuronsNum(),la.getPads());
				}
				catch(Exception e) {
					System.err.println(e.getMessage());
				}
			}
		}
		network.setOutputLayer(network.getLayers().lastElement());
		//Network.addOutputLayer(network, "dataO");//(network.getLayers().lastElement());
        
		try {
		//TODO: Input data is set for CIFAR here. update this to reflect other datasets.
		
				network.setDataFormats(inputDataShapeExample);

		}
		catch(Exception e) {
			System.err.println(e.getMessage());
		}
		
		return network;
	}
	
	
	public static Network createNetworkFromGenotype(Genotype<LayerGene> gt)
	{ 
		Network network = null;
		ArrayList<LayerGene> layerGenes = new ArrayList<>();
		
		for (Iterator<Chromosome<LayerGene>> i = gt.iterator(); i.hasNext(); ) 
		{
			AbstractChromosome<LayerGene> ac = (AbstractChromosome<LayerGene>)(i.next());
			LayerGene firstGene = ac.getGene(0);
			firstGene.getLayer().setName("f1" + firstGene.getLayer().getName());
			
			for (Iterator<LayerGene> j = ac.iterator(); j.hasNext(); ) 
			{
				LayerGene lg = j.next();
				if(lg.getLayer().getName().startsWith("skip"))
				{
					lg.getLayer().setNeuronsNum(ac.getGene(ac.length()-4).getLayer().getNeuronsNum()); // from reverse - add, skip, relu, conv. totalLen-4 is conv
					layerGenes.add(layerGenes.size()-1, lg);
				}
				else if(lg.getLayer().getName().startsWith("add"))
					layerGenes.add(layerGenes.size()-1, lg);
				else
					layerGenes.add(lg);
			}
		}
		network =createNetworkFromLayerGenes(layerGenes);
		
		//Tensor inputDataShapeExample = new Tensor(32,32,3);
		//network.setDataFormats(inputDataShapeExample);
		
		return network;
	}
}
