package nl.uva.aloha;

import java.util.ArrayList;
import java.util.Date;
import java.util.DoubleSummaryStatistics;
import java.util.Iterator;
import java.util.List;
import java.util.concurrent.Executors;
import java.util.stream.Collectors;

import espam.datamodel.graph.cnn.Network;
import io.jenetics.Genotype;
import io.jenetics.Optimize;
import io.jenetics.Phenotype;
import io.jenetics.engine.Engine;
import io.jenetics.engine.EvolutionResult;
import io.jenetics.ext.moea.MOEA;
import io.jenetics.ext.moea.NSGA2Selector;
import io.jenetics.ext.moea.Vec;
import io.jenetics.util.ISeq;
import io.jenetics.util.Seq;
import nl.uva.aloha.Alterers.GenotypeRepeater;
import nl.uva.aloha.Alterers.LayerGeneMutator;
import nl.uva.aloha.Alterers.NetworkRecombinator;
import nl.uva.aloha.Selectors.NSGA2CustomSelector;
import nl.uva.aloha.converters.GeneToNetwork;
import nl.uva.aloha.genetic.LayerGene;
import nl.uva.aloha.problems.PAMAP2Problem;
import nl.uva.aloha.helpers.Configs;

import nl.uva.aloha.helpers.OnnxHelper;
import nl.uva.aloha.helpers.OnnxGarbageCollector;
import nl.uva.aloha.helpers.OnnxRegistry;

public class MultiObjGAMain 
{
	
	/*static final int populationSize = 60;
	static final int numOfIterations = 80;
	static final int parallelism = 2; 
	static final int bestSurvivors = 1; 
	static final IntRange paretoSetSize = IntRange.of(10, 30);
	public static final boolean BatchNormAfterConv = true;
	public static final boolean seperableConv = false;*/
	
	static final Engine<LayerGene, Vec<double[]>> MULTI_ENGINE = 
			Engine.builder(new PAMAP2Problem())
			.populationSize(Configs.populationSize)
			.minimizing()
			.survivorsSize(0) //survivors are from offsprings as genotype repeater need to update EVERY survivor, to be able to train further.
			.offspringSelector(NSGA2CustomSelector.ofVec(Configs.bestSurvivors))
			.alterers(new NetworkRecombinator<Vec<double[]>>(0.3),
					  new LayerGeneMutator<Vec<double[]>>(0.3,0.12),
					  new GenotypeRepeater<Vec<double[]>>())
			.genotypeValidator( gt -> (MultiObjGAMain.isGenotypeValid(gt)) )
			.executor(Executors.newFixedThreadPool(Configs.parallelism))
			.build();
	
	
	public static void main(final String[] args) 
	{
		System.out.println("iterations - " + Configs.numOfGAIterations);
		
		Configs.DATASET = Configs.PAMAPDATASET;
		
		final ISeq<Phenotype<LayerGene, Vec<double[]>>> paretoset  = MULTI_ENGINE.stream()//new InitialPopulationCreator(populationSize).createPopulation() )
				.limit(Configs.numOfGAIterations)
				.peek(MultiObjGAMain::update)
				.collect(MOEA.toParetoSet(Configs.paretoSetSize));

			System.out.println("Best results are ::  "+ new Date().toString());
			
						
			Iterator<Phenotype<LayerGene, Vec<double[]>>> it = paretoset.iterator();
			while(it.hasNext())
			{
				Phenotype<LayerGene, Vec<double[]>> pt = it.next();
				String onnxName = OnnxRegistry.getInstance().getEntry(System.identityHashCode(pt.getGenotype()));
				System.out.println(pt.getGeneration() + " : " + onnxName + " : " + pt.getFitness());
			}
			 
	}

	public static Long generationTrack = (long)1;
	private static Double bestAccuracyYet = 0.0;
	private static List<Phenotype<LayerGene, Vec<double[]>>> bestPopulationYet = new ArrayList<>();
	
	private static void update (final EvolutionResult<LayerGene, Vec<double[]>> result)
	{
		//System.out.println("gen:" + result.getGeneration());
		DoubleSummaryStatistics stats = result.getPopulation().stream().mapToDouble(pt->pt.getFitness().data()[0]).summaryStatistics();
		
		
		DoubleSummaryStatistics stats_second = result.getPopulation().stream().mapToDouble(pt->pt.getFitness().data()[1]).summaryStatistics();
		
		System.out.println("gen: " + result.getGeneration() + "stats: " + stats.toString() + "second: " + stats_second.toString() );
		
		Configs.generationTrack = new Long(result.getGeneration() + 1);
			

		double bestOfThisEvolution = result.getBestFitness().data()[0];
		bestAccuracyYet = (bestOfThisEvolution > bestAccuracyYet)? bestOfThisEvolution : bestAccuracyYet;
		
		ISeq<Phenotype<LayerGene, Vec<double[]>>> paretoTemp =  NSGA2Selector.ofVec().select((Seq)result.getPopulation().asMSeq(), Configs.paretoSetSize.getMax(), Optimize.MINIMUM);
		
		Iterator<Phenotype<LayerGene, Vec<double[]>>> it = paretoTemp.iterator();
		while(it.hasNext())
		{
			Phenotype<LayerGene, Vec<double[]>> pt = it.next();
			String testOnnxName = OnnxRegistry.getInstance().getEntry(System.identityHashCode(pt.getGenotype()));
			System.out.println(pt.getGeneration() + " : " + testOnnxName + " : " +  pt.getFitness()+ ":" + OnnxHelper.testOnnx(testOnnxName));
			OnnxGarbageCollector.getInstance().removeFileFromDeletionList((int)(result.getGeneration()),
					OnnxRegistry.getInstance().getEntry(System.identityHashCode(pt.getGenotype())));

		}
		
		
		
		bestPopulationYet.addAll(result.getPopulation().stream().collect(Collectors.<Phenotype<LayerGene, Vec<double[]>>>toList()));
		bestPopulationYet = bestPopulationYet.stream().filter(pt->pt.getFitness().data()[0] > (bestAccuracyYet-0.01)).collect(Collectors.<Phenotype<LayerGene, Vec<double[]>>>toList());
		
		/*//AFTER LAST ITERATION
		if(result.getGeneration() == Configs.numOfGAIterations)
		{
			Iterator<Phenotype<LayerGene, Vec<double[]>>> it = result.getPopulation().iterator();
			while(it.hasNext())
			{
				Phenotype<LayerGene, Vec<double[]>> pt = it.next();
				testOnnxName = OnnxRegistry.getInstance().getEntry(System.identityHashCode(pt.getGenotype()));
				System.out.println(pt.getGeneration() + " : " + testOnnxName + " : " + 
									pt.getFitness()+ ":" + OnnxHelper.testOnnx(testOnnxName));
			}
			
		}*/
		
		
		
	}
	

	private static Boolean isGenotypeValid(Genotype<LayerGene> gt)
	{
		if(!( gt ).isValid())
			return false;
		
		Network network = GeneToNetwork.createNetworkFromGenotype(gt);
		if(network == null) 
			return false;
		
		if (network.checkConsistency())
			return true;
		else return false;
		
	}
	
}
