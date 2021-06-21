package nl.uva.aloha.genetic;

import java.util.UUID;

import espam.datamodel.graph.cnn.BoundaryMode;
import espam.datamodel.graph.cnn.Layer;
import espam.datamodel.graph.cnn.neurons.cnn.Convolution;
import espam.datamodel.graph.cnn.neurons.neurontypes.NonLinearType;
import espam.datamodel.graph.cnn.neurons.simple.NonLinear;
import io.jenetics.util.IntRange;
import io.jenetics.util.RandomRegistry;

@Deprecated
public class DualLayerGene 
	extends LayerGene

{
	private Layer _nextLayer; //1st layer is inherited from LayerGene class. We need only the nextLayer additional

	
	private static final long serialVersionUID = 9122135619559863698L;
	
	
	
	public DualLayerGene()
	{
		this(5, RandomRegistry.getRandom().nextInt(9)+1 );
	}
	
	public DualLayerGene(int numNuerons)
	{
		this(5,numNuerons);
	}
	
	public DualLayerGene(Layer convLayer) throws Exception
	{
		if(!(convLayer.getNeuron() instanceof Convolution))
			throw new Exception("Layer is not a convolution layer. Cannot initiate ConvReluGene from this layer");
		
		else 
		{
			_layer = (new Layer(convLayer));
			
			NonLinear relu = new NonLinear(NonLinearType.ReLU);
			setNextLayer(new Layer("relu"+UUID.randomUUID().toString().replace("-", ""),relu,convLayer.getNeuronsNum()));			
		}
	}
	
	public DualLayerGene(int kernelSize,int numNuerons)
	{
		Convolution cnv = new Convolution(kernelSize, BoundaryMode.VALID); //KernelSize, boundary, stride(default=1)
		_layer = (new Layer("conv"+UUID.randomUUID().toString().replace("-", ""),cnv, numNuerons));
		
		NonLinear relu = new NonLinear(NonLinearType.ReLU);
		_nextLayer = (new Layer("relu"+UUID.randomUUID().toString().replace("-", ""),relu,numNuerons));
		
	}
	
	public static DualLayerGene of(final int kernelSize,final int maxNumNeurons, int minNumNuerons)
	{
		return new DualLayerGene(kernelSize,RandomRegistry.getRandom().nextInt(maxNumNeurons-minNumNuerons)-minNumNuerons);
	}
	
	
	public static DualLayerGene of(final int kernelSize,final IntRange range)
	{
		int maxNumNeurons = range.getMax();
		int minNumNuerons = range.getMin();
		return new DualLayerGene(kernelSize,RandomRegistry.getRandom().nextInt(maxNumNeurons-minNumNuerons)+minNumNuerons);
	}
	
	public static DualLayerGene of(final int kernelSize,final int numNuerons)
	{
		return new DualLayerGene(kernelSize,numNuerons);
	}
	
	public DualLayerGene newInstance()
	{
		return new DualLayerGene();
	}
	
	@Override
	public DualLayerGene newInstance(Layer convLayer) 
	{
		try {
			return new DualLayerGene(convLayer);
		}
		catch (Exception e) {
			return null;
		}
	}
	

	public Layer getNextLayer()
	{
		return _nextLayer;
	}

	public void setNextLayer(Layer reluLayer) {
		this._nextLayer = reluLayer;
	}

	@Override
	public boolean isValid() 
	{
			
		return ( (_layer!=null) && (_nextLayer!=null) && (_layer.getNeuronsNum() == _nextLayer.getNeuronsNum()) );
	}


	
	
}
