package nl.uva.aloha.genetic;

import java.io.Serializable;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Base64;
import java.util.Base64.Decoder;

import org.apache.commons.math3.genetics.RandomKeyMutation;

import java.util.UUID;

import espam.datamodel.graph.cnn.BoundaryMode;
import espam.datamodel.graph.cnn.Layer;
import espam.datamodel.graph.cnn.Neuron;
import espam.datamodel.graph.cnn.neurons.arithmetic.Arithmetic;
import espam.datamodel.graph.cnn.neurons.cnn.Convolution;
import espam.datamodel.graph.cnn.neurons.cnn.Pooling;
import espam.datamodel.graph.cnn.neurons.neurontypes.ArithmeticOpType;
import espam.datamodel.graph.cnn.neurons.neurontypes.DataType;
import espam.datamodel.graph.cnn.neurons.neurontypes.NonLinearType;
import espam.datamodel.graph.cnn.neurons.neurontypes.PoolingType;
import espam.datamodel.graph.cnn.neurons.simple.Data;
import espam.datamodel.graph.cnn.neurons.simple.DenseBlock;
import espam.datamodel.graph.cnn.neurons.simple.NonLinear;
import espam.datamodel.graph.cnn.neurons.transformation.Reshape;
import espam.datamodel.graph.cnn.neurons.transformation.Transpose;
import espam.datamodel.graph.csdf.datasctructures.Tensor;
import io.jenetics.Gene;
import io.jenetics.util.RandomRegistry;
import nl.uva.aloha.GAMain;
import nl.uva.aloha.helpers.Configs;

public class LayerGene 
		implements Gene<Layer,LayerGene>, Serializable
{
	
	private static final long serialVersionUID = 8077673452944666485L;

	protected Layer _layer;
	protected int _minNumNeurons = 1;
	protected int _maxNumNeurons = 20; //Default values if not set explicitly.//TODO: Dense block should have a lot more number of neurons
	protected String _layerType;
	static protected int discreteLevel = Configs.discreteLevel; 
	//@SuppressWarnings("unused")
	//private ConnectionType _cnxnTypeToNextLayer;
	
	
	private static ArrayList<String> LayerTypes = new ArrayList<String>(Arrays.asList( "Convolution", 
															                    "Pooling",
															                    "GlobalAveragePool",
															                    "NonLinear",
															                    "DenseBlock",
															                    "DeConv",
															                    "DataI",
															                    "DataO"));
			
	public LayerGene(Layer l)
	{
		_layer = l; //TODO : cloning throws error - check what is going on
		_layerType = l.getNeuron().getNeuronType().toString();
		_maxNumNeurons = l.getNeuronsNum()+1;
		_minNumNeurons = 1;
		
		//_cnxnTypeToNextLayer = cnxn; //ConnectionType is enum - so no need to clone. 
	}
	
	
	public LayerGene(String layerType)
	{
		_layerType = layerType;
		_layer = createLayerOfType(layerType, getRandomNumNeuron(1,20));
	}
	
	public LayerGene(String layerType,int numberOfNeurons)
	{
		_layerType = layerType;
		_layer = createLayerOfType(layerType, numberOfNeurons);
	}
	
	public LayerGene(String layerType,int minNeurons, int maxNeurons)
	{
		_layerType = layerType;
		_maxNumNeurons = maxNeurons;
		_minNumNeurons = minNeurons;
		_layer = createLayerOfType(layerType, getRandomNumNeuron(minNeurons,maxNeurons));
	}
	
	public static int getRandomNumNeuron(int minNeurons, int maxNeurons)
	{
		if(minNeurons >= maxNeurons)
			return maxNeurons;
		
		int neuronNum =  RandomRegistry.getRandom().nextInt(maxNeurons - minNeurons) + minNeurons; 
		return (Math.max(1, Math.round(neuronNum/discreteLevel)*discreteLevel));
	}
	
	
	public LayerGene(Layer l,int minNeurons, int maxNeurons)
	{
		_layer = l; //TODO : cloning throws error - check what is going on - UPDATE: cloned layer is being sent here now
		_layerType = l.getNeuron().getNeuronType().toString();
		_maxNumNeurons = maxNeurons;
		_minNumNeurons = minNeurons;
	}
	
	public LayerGene()
	{
		String randomLayerType = LayerTypes.get(RandomRegistry.getRandom().nextInt(LayerTypes.size()-1)); // -1 to size to avoid a new data layer
		_layerType = randomLayerType;
		_layer = createLayerOfType(randomLayerType, getRandomNumNeuron(1,20));
	}
	
	
	
	private Layer createLayerOfType(String layerType, int numberOfNeurons) 
	{
		Layer l = null;
		switch (layerType)
		{
			case "dataI":  	Data inputData = new Data(DataType.INPUT);
						   	l = new Layer("input",inputData,1);
				break;
				
			case "dataO":  	Data outData = new Data(DataType.OUTPUT);
			   				l = new Layer("output",outData,1);
			   	break;
	
			case "Convolution": 
							if(Configs.DATASET.equals("PAMAP2"))
							{
								Convolution cnv = new Convolution(1,5, BoundaryMode.VALID); //KernelSize, boundary, stride(default=1)
								//l = new Layer("conv"+UUID.randomUUID().toString().replace("-", ""),cnv,numberOfNeurons);// +1 because it generates 0 as well
								l = new Layer("conv"+shortUUID(),cnv,numberOfNeurons);// +1 because it generates 0 as well
								
								l.setPads(2,0,2,0);
							}
							
							
							else
							{
								Convolution cnv = new Convolution(3, BoundaryMode.VALID); //KernelSize, boundary, stride(default=1)
								l = new Layer("conv"+shortUUID(),cnv,numberOfNeurons);// +1 because it generates 0 as well
								l.setPads(1,1,1,1);
							}
							//_cnxnTypeToNextLayer = ConnectionType.ONETOALL;
				break;
			case "Convolution1K":
							Convolution cnv1k = new Convolution(1, BoundaryMode.VALID); //KernelSize, boundary, stride(default=1)
							l = new Layer("conv"+shortUUID(),cnv1k,numberOfNeurons);// +1 because it generates 0 as well
							;
				break;
			case "deConv":
				     		Convolution  deConv = new Convolution(3, BoundaryMode.VALID,1,true);
				     		l = new Layer("deconv"+shortUUID(),deConv,numberOfNeurons);// +1 because it generates 0 as well
							l.setPads(1,1,1,1);//TODO: check if padding is needed for deconvolution!
			
				break;
			case "Pooling": 
							Pooling pool = new Pooling(PoolingType.MAXPOOL,2);
							//if(GAMain.DATASET.equals("PAMAP2"))
							//{
							//	pool = new Pooling(PoolingType.MAXPOOL,2,1,BoundaryMode.VALID, 2); //Stride=2
							//}
							l = new Layer("pool"+shortUUID(),pool,numberOfNeurons);
				break;
			case "MaxP1": 
							Pooling pool1 = new Pooling(PoolingType.MAXPOOL,2,2,BoundaryMode.VALID,1);
							l = new Layer("mxpool"+shortUUID().substring(0, 4),pool1,numberOfNeurons);
							l.setPads(0,0,1,1);
				break;
	
			case "GlobalMaxPool": Pooling mapool = new Pooling(PoolingType.GLOBALMAXPOOL); //Stride, kernel etc doesnt matter
							l = new Layer("gapool"+shortUUID(),mapool,numberOfNeurons);
				break;
			case "GlobalAveragePool": Pooling gapool = new Pooling(PoolingType.GLOBALAVGPOOL,1); //Stride=1
							l = new Layer("gapool"+shortUUID(),gapool,numberOfNeurons);
				break;
		
			case "NonLinear": NonLinear relu = new NonLinear(NonLinearType.ReLU);
							l = new Layer("relu"+shortUUID(),relu,numberOfNeurons);
				break;
			case "LeakyRelu": NonLinear leakyrelu = new NonLinear(NonLinearType.LeakyReLu);
							l = new Layer("leakyrelu"+shortUUID(),leakyrelu,numberOfNeurons);
				break;
				
			case "DenseBlock": DenseBlock dense = new DenseBlock(numberOfNeurons);
							l = new Layer("dense"+shortUUID(), dense,1);
				break;
				
			case "Softmax":  NonLinear softmax = new NonLinear( NonLinearType.SOFTMAX);
							l = new Layer("softmax", softmax,numberOfNeurons);
				break;
				
			case "TransposeTY": Transpose transp = new Transpose();
								//transp.
								 l = new Layer("Transpose", transp,1);
				break;
			case "ReshapeTY": Reshape reshp = new Reshape(new Tensor(12,12,125), new Tensor(12,12,5,25));
							l = new Layer("reshape", reshp,1);
				break;
			case "Add":  	Arithmetic add = new Arithmetic(ArithmeticOpType.ADD);
							l = new Layer("add"+shortUUID(), add,1);
				break;
			case "Skip":    Convolution skp = new Convolution(1, BoundaryMode.VALID); //KernelSize, boundary, stride(default=1)
							l = new Layer("skip"+shortUUID(),skp,numberOfNeurons);// +1 because it generates 0 as well
							l.setPads(0,0,0,0);
				break;
		}
		
		return l;
	}

	
	public LayerGene clone()
	{
		Layer newLayer = new Layer(getLayer().getName(), Neuron.copyNeuron(getLayer().getNeuron()), getLayer().getNeuronsNum());
		if(getLayer().getPads()!=null)
			newLayer.setPads(getLayer().getPads().clone());
		return new LayerGene(newLayer,_minNumNeurons,_maxNumNeurons);
	}
	
	public LayerGene mutate(double rate)
	{
		//Number of neurons change here - 
		LayerGene lg = this.clone();
		Float randomNextProb = RandomRegistry.getRandom().nextFloat();
		if(lg.getLayer().getNeuron() instanceof Convolution)
		{
			if(lg.getLayer().getName().contains("skip"))
				return lg;
			if(randomNextProb<=0.8)
			{
				int originalNeuronsNum = lg._layer.getNeuronsNum();
				//If rate = 0.1 We want final rate to be between 0.9 and 1.1 to reflect 10% up or down changes
				// 1.0 + rate - random number between <0.0,1.0> * rate*2 = 1.1 - random number between 0.0 and 0.2. 
				
				float finalRate = (float)(1.0 + rate - (RandomRegistry.getRandom().nextDouble()*rate*2));
				int modifiedNeuronsNum = Math.round(originalNeuronsNum*finalRate);
				modifiedNeuronsNum = discreteLevel*Math.round(modifiedNeuronsNum/discreteLevel);
				
				lg._layer.setNeuronsNum(Math.max(modifiedNeuronsNum,_minNumNeurons));
			}
			else //if((randomNextProb>0.8)&&(randomNextProb<0.97))) //17% of cases change kernel size
			{
				Convolution cnv = (Convolution)lg.getLayer().getNeuron();
				int k = cnv.getKernelW();
				
				if((randomNextProb<0.9) && (k<7))
				{
					int newk = k+2;
					int newp = Math.floorDiv(newk, 2);
					
					if(Configs.DATASET.equals("PAMAP2"))
					{
						cnv.setKernelW(newk);
						lg.getLayer().setPads(newp,0,newp,0);
					}
					else
					{
						cnv.setKernelSize(newk);
						lg.getLayer().setPads(newp,newp,newp,newp);
					}
					
				}
				else if((randomNextProb>=0.9) && (k>4))
				{
					int newk = k-2;
					int newp = Math.floorDiv(newk, 2);
					
					if(Configs.DATASET.equals("PAMAP2"))
					{
						cnv.setKernelW(newk);
						lg.getLayer().setPads(newp,0,newp,0);
					}
					else
					{
						cnv.setKernelSize(newk);
						lg.getLayer().setPads(newp,newp,newp,newp);
					}
					
				}
			}
			
		}
		else if((lg.getLayer().getNeuron() instanceof NonLinear) && !(lg.getLayer().getNeuron().getName().contains(NonLinearType.SOFTMAX.toString())))
		{
						
		}
		else if(lg.getLayer().getNeuron() instanceof DenseBlock)
		{
			DenseBlock db = (DenseBlock)lg._layer.getNeuron();
			int originalNeuronsNum = db.getNeuronsNum();
			float finalRate = (float)(1.0 + rate - (RandomRegistry.getRandom().nextDouble()*rate*2));
			int modifiedNeuronsNum = Math.round(originalNeuronsNum*finalRate);
			modifiedNeuronsNum = Math.max(discreteLevel*Math.round(modifiedNeuronsNum/discreteLevel),_minNumNeurons);
			db.setNeuronsNum(modifiedNeuronsNum);
		}
		
		/*else if(lg.getLayer().getNeuron() instanceof Pooling)
		{
			int typeNums = new Float(PoolingType.values().length*randomNextProb).intValue();
			((Pooling)lg.getLayer()).setNeuronType(PoolingType.values()[typeNums]);
		}*/
		
		
		return lg;
		
	}
	public void changeLayerSize(int neuronsNum)
	{
		_layer.setNeuronsNum(neuronsNum);
	}
	
	private String shortUUID() 
	{
		UUID uuid = UUID.randomUUID();
		ByteBuffer byteBuffer = ByteBuffer.allocate(16);
		 byteBuffer.putLong(uuid.getMostSignificantBits());
		 byteBuffer.putLong(uuid.getLeastSignificantBits());
		 
		 String shortuuid = Base64.getEncoder().withoutPadding().encodeToString(byteBuffer.array()).replaceAll("[^a-zA-Z\\d]", "");
		
		 return shortuuid;
		
	}
	
	//return Base64.getEncoder().withoutPadding().encodeToString(byteBuffer.array())
    //.replaceAll(&quot;/&quot;, &quot;_&quot;)
    //.replaceAll(&quot;\\+&quot;, &quot;-&quot;);
	
	
	/*public void changeKernel(int kernelSize)
	{
		changeKernel(kernelSize, BoundaryMode.VALID, 1);
	}
	
	public void changeKernel(int kernelSize, int kernelStride)
	{
		changeKernel(kernelSize, BoundaryMode.VALID, kernelStride);
	}
	
	public void changeKernelStride(int kernelStride)
	{
		changeKernel(_layer.getNeuron().get, BoundaryMode.VALID, kernelStride);
	}
	
	public void changeKernel(int kernelSize, BoundaryMode boundaryMode)
	{
		changeKernel(kernelSize, boundaryMode, 1);
	}
	
	public void changeKernel(int kernelSize,BoundaryMode boundaryMode,int kernelStride)
	{
		if(_layer.getNeuron() instanceof Convolution)
		{
			_layer.setNeuron(new Convolution(kernelSize,boundaryMode,kernelStride));
		}
	}
	*/
	/*public void changeConnectionTypeTo(ConnectionType connectionType)
	{
		_cnxnTypeToNextLayer = connectionType;
	}*/
	
	
	
	
	@Override
	public String toString() {
		return _layer.toString() + " : {" + _layer.getNeuron().toString() + "}";
	}

	@Override
	public boolean isValid() 
	{		
		return (_layer!=null);
	}

	@Override
	public Layer getAllele() 
	{
		return _layer;
	}

	public Layer getLayer() 
	{
		return _layer;
	}
	
	@Override
	public LayerGene newInstance() 
	{
		return new LayerGene();
	}

	@Override
	public LayerGene newInstance(Layer layer) 
	{
	
		return new LayerGene(layer);
	}

	public int getMinNumNeurons() {
		return _minNumNeurons;
	}

	public void setMinNumNeurons(int minNumNeurons) {
		_minNumNeurons = minNumNeurons;
	}

	public int getMaxNumNeurons() {
		return _maxNumNeurons;
	}

	public void setMaxNumNeurons(int maxNumNeurons) {
		_maxNumNeurons = maxNumNeurons;
	}

	public String getLayerType() {
		return _layerType;
	}

	public void setLayerType(String _layerType) {
		this._layerType = _layerType;
	}
	
}
