package nl.uva.aloha.helpers;

import java.io.FileReader;
import java.io.Reader;
import java.util.ArrayList;
import java.util.Iterator;

import espam.datamodel.graph.csdf.datasctructures.Tensor;
import io.jenetics.Chromosome;
import io.jenetics.Genotype;
import io.jenetics.util.IntRange;
import nl.uva.aloha.genetic.DualLayerChromosome;
import nl.uva.aloha.genetic.LayerGene;
import nl.uva.aloha.genetic.SimpleLayerChromosome;

public class Configs {

	private static Configs singleton = null;
	
	public static String projectId = "a1";
	public static  int populationSize = 10;
	public static  int numOfGAIterations = 80;
	public static  int parallelism = 1; //depends on resources available - GPUs etc. 
	public static  int bestSurvivors = 2; 
	public static  IntRange paretoSetSize = IntRange.of(10, 15);
	public static final boolean BatchNormAfterConv = false; //Automatically adds batchnorm layer after convolution
	public static  int discreteLevel = 16;
	public static double mutationRate = 0.3;
	public static double crossoverRate = 0.3;
	
	public static int NUM_OUTPUT_CLASSES = 10;
	static public Tensor INPUT_DATA_CIFAR_SHAPE = new Tensor(32,32,3);
	static public Tensor INPUT_PAMAP2_SLIDING_SHAPE = new Tensor(100,1,40);
	static public Tensor INPUT_DATA_SHAPE = null;
	static public Tensor OUTPUT_DATA_SHAPE = null;
	
	public static Long generationTrack = (long)1;
	
	static public int deleteOnnxAfterGens = 3;

	//PATHS ---- 
	//public static String OnnxWorkspacePath = "/home/dsapra/aloha/onnx/";
	//public static String PythonPath = "/usr/bin/python3";
	
	public static String OnnxWorkspacePath = "/Users/sne/aloha_workspace/onnx/";
	public static String PythonPath = "/usr/local/bin/python3";
	
	public static final String CIFARDATASET = "CIFAR";
	public static final String PAMAPDATASET = "PAMAP2";
	public static final String VOCDATASET = "VOC";
	
	//public static final String DATASET = "TINYYOLO";
	
	public static String DATASET = CIFARDATASET;
	
	
	public static final boolean seperableConv = false;
	
	
    //public static int serverPollingIntervalMinutes = 1; //minutes
	
	public static Genotype<LayerGene> ENCODING;
    
    
    public static Configs getSingleton()
    {
    	if(singleton == null)
    		initialize();
    	return singleton;
    }
    
    public static void initialize()
    {
    	if(singleton == null)
    		singleton = new Configs();
    }
    
	
	public Configs getConfigs()
	{
		return singleton;
	}
	
	public static void main(final String[] args) 
	{
	
		Configs.initialize();
	}
	
	
	
}
