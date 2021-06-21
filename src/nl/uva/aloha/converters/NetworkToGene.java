package nl.uva.aloha.converters;

import java.util.ArrayList;
import java.util.List;

import espam.datamodel.graph.cnn.Layer;
import espam.datamodel.graph.cnn.Network;
import nl.uva.aloha.genetic.LayerGene;

public class NetworkToGene {

	
	public static List<LayerGene> extractLayerGenesFromNetwork(Network network)
	{
		List<LayerGene> layers = new ArrayList<LayerGene>();
		int i;
		for(i=0;i<network.getLayers().size()-1;i++) 
		{
			Layer l = network.getLayers().get(i);
			//layers.add(new LayerGene(l, network.getConnections().get(i).getType()));
		}
		//layers.add(new LayerGene(network.getLayers().get(i), null));
		
		return layers;
	}
	
}
