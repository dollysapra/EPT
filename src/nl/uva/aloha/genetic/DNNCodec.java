package nl.uva.aloha.genetic;

import java.util.function.Function;

import io.jenetics.Genotype;
import io.jenetics.engine.Codec;
import io.jenetics.util.Factory;
import io.jenetics.util.IntRange;
import nl.uva.aloha.helpers.Configs;

public class DNNCodec implements Codec<GenotypeTranslation, LayerGene>
{

	public static final Genotype<LayerGene> ENCODING = Genotype.of(
			SimpleLayerChromosome.of("dataI",1,2),
			DualLayerChromosome.of("Convolution:NonLinear", 32, 64,IntRange.of(4,7)), //2-3// Min is inclusive, Max is exclusive for range
			SimpleLayerChromosome.of("Pooling",5,50),
			DualLayerChromosome.of("Convolution:NonLinear", 32, 128,IntRange.of(6,9)),//3-4
			SimpleLayerChromosome.of("Pooling",5,50),
			DualLayerChromosome.of("Convolution:NonLinear", 288, 640,IntRange.of(6,15)),// 3-7
			SimpleLayerChromosome.of("Pooling",5,50),
			//SimpleLayerChromosome.of("GlobalMaxPool",5,50),
			DualLayerChromosome.of("DenseBlock:NonLinear",256,1024,IntRange.of(4, 7)),
			//SimpleLayerChromosome.of("DenseBlock",100,900,IntRange.of(2, 4)),

			SimpleLayerChromosome.of("DenseBlock",10,10), //LAST layer is fully connected layer, num of neurons = number of output classes
			SimpleLayerChromosome.of("Softmax",10,10),
			SimpleLayerChromosome.of("dataO",1,2)
		);

	public static final Genotype<LayerGene>  CIFAR_ENCODING = Genotype.of(
			SimpleLayerChromosome.of("dataI",1,2),
			
			DualLayerChromosome.of("Convolution:NonLinear", 32, 64,IntRange.of(2,3)),         // 2-3 //Min is inclusive, Max is exclusive for range
			SkipCnxnDualLayerChromosome.of("Convolution:NonLinear", 32, 64,IntRange.of(4,9)), // 1-3 //Min is inclusive, Max is exclusive for range
			SimpleLayerChromosome.of("Pooling",5,50),
			
			SkipCnxnDualLayerChromosome.of("Convolution:NonLinear", 64, 128,IntRange.of(4,9)), //1-3
			//SkipCnxnDualLayerChromosome.of("Convolution:NonLinear", 64, 256,IntRange.of(4,9)),//1-3 //Min is inclusive, Max is exclusive for range
			//impleLayerChromosome.of("Pooling",5,50),
			
			//SkipCnxnDualLayerChromosome.of("Convolution:NonLinear", 128, 256,IntRange.of(4,11)), //1-4 //Min is inclusive, Max is exclusive for range
			//SkipCnxnDualLayerChromosome.of("Convolution:NonLinear", 128, 512,IntRange.of(4,11)),//1-4
			SimpleLayerChromosome.of("Pooling",5,50),
			
			
			DualLayerChromosome.of("DenseBlock:NonLinear",256,1024,IntRange.of(4, 7)),
			
			SimpleLayerChromosome.of("DenseBlock",10,10), //LAST layer is fully connected layer, num of neurons = number of output classes
			SimpleLayerChromosome.of("Softmax",10,10),
			SimpleLayerChromosome.of("dataO",1,2)
		);
	
	public static final Genotype<LayerGene> PAMAP2ENCODING = Genotype.of(
			SimpleLayerChromosome.of("dataI",1,2),
			DualLayerChromosome.of("Convolution:NonLinear", 64,128,IntRange.of(4,9)), // 2-4 //Min is inclusive, Max is exclusive for range
			SimpleLayerChromosome.of("Pooling",5,50),
			DualLayerChromosome.of("Convolution:NonLinear", 96, 256,IntRange.of(4,11)),//2-5
			//SimpleLayerChromosome.of("Pooling",5,50),
			SimpleLayerChromosome.of("GlobalMaxPool",5,50),
			DualLayerChromosome.of("DenseBlock:NonLinear",128,512,IntRange.of(2, 7)),
			SimpleLayerChromosome.of("DenseBlock",12,12), //LAST layer is fully connected layer, num of neurons = number of output classes
			SimpleLayerChromosome.of("Softmax",12,12),
			SimpleLayerChromosome.of("dataO",1,2)
		);
	
	public static final Genotype<LayerGene> HW2ENCODING = Genotype.of(
			SimpleLayerChromosome.of("dataI",1,2),
			DualLayerChromosome.of("Convolution:NonLinear", 32, 96,IntRange.of(4,7)), // 2-4 //Min is inclusive, Max is exclusive for range
			SimpleLayerChromosome.of("Pooling",5,50),
			DualLayerChromosome.of("Convolution:NonLinear", 128, 192,IntRange.of(4,7)),//2-5
			//SimpleLayerChromosome.of("Pooling",5,50),
			SimpleLayerChromosome.of("GlobalMaxPool",5,50),
			DualLayerChromosome.of("DenseBlock:NonLinear",200,400,IntRange.of(2, 5)), // 1-2
			SimpleLayerChromosome.of("DenseBlock",3,3), //LAST layer is fully connected layer, num of neurons = number of output classes
			SimpleLayerChromosome.of("Softmax",3,3),
			SimpleLayerChromosome.of("dataO",1,2)
		);
	
	
	public static final Genotype<LayerGene> TINYENCODING = Genotype.of(
			SimpleLayerChromosome.of("dataI",1,2),
			DualLayerChromosome.of("Convolution:LeakyRelu", 16, 16,IntRange.of(2,3)), 
			SimpleLayerChromosome.of("Pooling",5,50),
			DualLayerChromosome.of("Convolution:LeakyRelu", 32, 32,IntRange.of(2,3)),
			SimpleLayerChromosome.of("Pooling",5,50),
			DualLayerChromosome.of("Convolution:LeakyRelu", 64, 64,IntRange.of(2,3)), 
			SimpleLayerChromosome.of("Pooling",5,50),
			DualLayerChromosome.of("Convolution:LeakyRelu", 128, 128,IntRange.of(2,3)), 
			SimpleLayerChromosome.of("Pooling",5,50),
			DualLayerChromosome.of("Convolution:LeakyRelu", 256, 256,IntRange.of(2,3)), 
			SimpleLayerChromosome.of("Pooling",5,50),
			DualLayerChromosome.of("Convolution:LeakyRelu", 512, 512,IntRange.of(2,3)), 
			SimpleLayerChromosome.of("MaxP1",50,50),
			DualLayerChromosome.of("Convolution:LeakyRelu", 1024, 1024,IntRange.of(2,3)),
			SimpleLayerChromosome.of("Convolution1K",100,100),
			//SimpleLayerChromosome.of("TransposeTY",1,1),
			//SimpleLayerChromosome.of("ReshapeTY",1,1),
			SimpleLayerChromosome.of("dataO",1,2)

			
			);	
	
	public static final Genotype<LayerGene>  VOCENCODING = Genotype.of(
			SimpleLayerChromosome.of("dataI",1,2),
			
			DualLayerChromosome.of("Convolution:NonLinear", 32, 96,IntRange.of(4,7)), // 1-2 //Min is inclusive, Max is exclusive for range
			SimpleLayerChromosome.of("Pooling",5,50),
			
			SkipCnxnDualLayerChromosome.of("Convolution:NonLinear", 32, 96,IntRange.of(4,13)), // 1-2 //Min is inclusive, Max is exclusive for range
			SimpleLayerChromosome.of("Pooling",5,50),
			
			SkipCnxnDualLayerChromosome.of("Convolution:NonLinear", 32, 96,IntRange.of(4,13)), // 1-2 //Min is inclusive, Max is exclusive for range
			SimpleLayerChromosome.of("Pooling",5,50),
			
			//SkipCnxnDualLayerChromosome.of("Convolution:NonLinear", 32, 96,IntRange.of(4,13)), // 1-2 //Min is inclusive, Max is exclusive for range
			//SimpleLayerChromosome.of("Pooling",5,50),
			
			SkipCnxnDualLayerChromosome.of("Convolution:NonLinear", 128, 192,IntRange.of(4,13)),//2-5
			SimpleLayerChromosome.of("Pooling",5,50),
			//SimpleLayerChromosome.of("GlobalAveragePool",5,50),
			
			SimpleLayerChromosome.of("DenseBlock",200,600,IntRange.of(2, 5)),
			SimpleLayerChromosome.of("DenseBlock",20,20), //LAST layer is fully connected layer, num of neurons = number of output classes
			
			SimpleLayerChromosome.of("dataO",1,2)
		);
	
	
	
	@Override
	public Function<Genotype<LayerGene>, GenotypeTranslation> decoder() 
	{
		return gt-> new GenotypeTranslation(gt);
	}

	@Override
	public Factory<Genotype<LayerGene>> encoding() {
		if(Configs.DATASET.equals("VOC"))
			return VOCENCODING;
		else if(Configs.DATASET.equals("PAMAP2"))
			return PAMAP2ENCODING;
		else if(Configs.DATASET.equals("CIFAR"))
			return CIFAR_ENCODING;
		return ENCODING;
	}

}

