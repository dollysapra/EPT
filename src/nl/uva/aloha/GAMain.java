package nl.uva.aloha;

import java.util.ArrayList;
import java.util.Date;
import java.util.DoubleSummaryStatistics;
import java.util.Iterator;
import java.util.List;
import java.util.concurrent.ForkJoinPool;
import java.util.stream.Collectors;

import espam.datamodel.graph.cnn.Network;
import io.jenetics.Genotype;
import io.jenetics.Phenotype;
import io.jenetics.TournamentSelector;
import io.jenetics.engine.Engine;
import io.jenetics.engine.EvolutionResult;
import io.jenetics.ext.moea.NSGA2Selector;
import io.jenetics.ext.moea.Vec;
import nl.uva.aloha.Alterers.GenotypeRepeater;
import nl.uva.aloha.Alterers.LayerGeneMutator;
import nl.uva.aloha.Alterers.NetworkRecombinator;
import nl.uva.aloha.Alterers.ONNXAlteration;
import nl.uva.aloha.Selectors.CustomTruncatationSelector;
import nl.uva.aloha.converters.GeneToNetwork;
import nl.uva.aloha.problems.AlohaProblem;
import nl.uva.aloha.genetic.DNNCodec;
import nl.uva.aloha.genetic.LayerGene;
import nl.uva.aloha.helpers.Configs;
import nl.uva.aloha.helpers.OnnxGarbageCollector;
import nl.uva.aloha.helpers.OnnxHelper;
import nl.uva.aloha.helpers.OnnxRegistry;

public class GAMain {

	
	
	
	static final Engine<LayerGene, Double> SINGLE_OBJECTIVE_ENGINE = 
			Engine.builder(GAMain::accuracyFitness,DNNCodec.PAMAP2ENCODING)
			.populationSize(Configs.populationSize)
			.maximizing()
			.survivorsSize(0)//GenotypeRepeater takes care of survivors and remaining population (OLD onnx versions have a bug in training the same file again)
			.offspringSelector(new CustomTruncatationSelector<>(Configs.populationSize-Configs.bestSurvivors))
			//.survivorsSelector(new TruncationSelector<>(bestSurvivors))
			.alterers(new NetworkRecombinator<Double>(0.4),
					  new LayerGeneMutator<Double>(0.4,0.12),
					  new GenotypeRepeater<Double>())
			.genotypeValidator( gt -> (GAMain.isGenotypeValid(gt)) )
			.executor(new ForkJoinPool(Configs.parallelism))
			.build();
		
	
	
	private static Double bestAccuracyYet = 0.0;
	private static List<Double> allResults = new ArrayList<>();
	public static Double rateOfChange = 1.0;
	private static List<Phenotype<LayerGene, Double>> bestPopulationYet = new ArrayList<>();
	private static List<Phenotype<LayerGene, Double>> bestofEachGen = new ArrayList<>();
	
	

	public static void main(final String[] args) 
	{
		System.out.println("iterations - " + Configs.numOfGAIterations);
		final EvolutionResult best = (EvolutionResult)SINGLE_OBJECTIVE_ENGINE.stream()//new InitialPopulationCreator(populationSize).createPopulation() )
				.limit(Configs.numOfGAIterations)
				.peek(GAMain::update)
				.collect(EvolutionResult.toBestEvolutionResult());

			System.out.println("Best result is "+best.getBestFitness() + "---" + new Date().toString());
			
			String bestONNX = OnnxRegistry.getInstance().getEntry(System.identityHashCode(best.getBestPhenotype().getGenotype()));
			ONNXAlteration onnxAlterGt = new ONNXAlteration(OnnxRegistry._onnxFolder+ bestONNX);
			
			onnxAlterGt.resetReshapeLayerDimForTesting();
			onnxAlterGt.updateONNXFile();
			
			
			System.out.println("ONNX: " + bestONNX);
			System.out.println("Best set:");
			
			Iterator<Phenotype<LayerGene, Double>> it = bestPopulationYet.iterator();
			while(it.hasNext())
			{
				Phenotype<LayerGene, Double> pt = it.next();
				String testOnnxName = OnnxRegistry.getInstance().getEntry(System.identityHashCode(pt.getGenotype()));
				System.out.println(pt.getGeneration() + " : " + testOnnxName + " : " + 
									pt.getFitness());//+ ":" + OnnxHelper.testOnnx(testOnnxName));
			}
			
			 
	}
	
	private static void update (final EvolutionResult<LayerGene, Double> result)
	{
		//System.out.println("gen:" + result.getGeneration());
		DoubleSummaryStatistics stats = result.getPopulation().stream().mapToDouble(pt->pt.getFitness()).summaryStatistics();
		DoubleSummaryStatistics topstats = result.getPopulation().stream().mapToDouble(pt->pt.getFitness()).map(d->1.0-d).sorted().map(d->1.0-d).
				limit(50).summaryStatistics();
		
		
		allResults.add(topstats.getAverage());
                if(allResults.size() >=3)
                {
                        rateOfChange = Math.abs(allResults.get(allResults.size()-1) - allResults.get(allResults.size()-3))*30;
                }

		System.out.println("gen: " + result.getGeneration() + "stats: " + stats.toString() + "top50 avg:" + topstats.getAverage() 
							+ "rateofchange" + rateOfChange );
		generationTrack = new Long(result.getGeneration() + 1);
		
		Double bestOfThisEvolution = result.getBestFitness();
		bestAccuracyYet = (bestOfThisEvolution > bestAccuracyYet)? bestOfThisEvolution : bestAccuracyYet;


		Phenotype<LayerGene, Double> pt1 = result.getBestPhenotype();
		String testOnnxName = OnnxRegistry.getInstance().getEntry(System.identityHashCode(pt1.getGenotype()));
		System.out.println(pt1.getGeneration() + " : " + testOnnxName + " : " +  pt1.getFitness()+ ":" + OnnxHelper.testOnnx(testOnnxName));
		
		OnnxGarbageCollector.getInstance().removeFileFromDeletionList((int)(result.getGeneration()),
							OnnxRegistry.getInstance().getEntry(System.identityHashCode(result.getBestPhenotype().getGenotype())));
		
		bestPopulationYet.addAll(result.getPopulation().stream().collect(Collectors.<Phenotype<LayerGene, Double>>toList()));
		bestPopulationYet = bestPopulationYet.stream().filter(pt->pt.getFitness() > (bestAccuracyYet-0.01)).collect(Collectors.<Phenotype<LayerGene, Double>>toList());
		
		if(result.getGeneration() == Configs.numOfGAIterations)
		{
			Iterator<Phenotype<LayerGene, Double>> it = result.getPopulation().iterator();
			while(it.hasNext())
			{
				Phenotype<LayerGene, Double> pt = it.next();
				testOnnxName = OnnxRegistry.getInstance().getEntry(System.identityHashCode(pt.getGenotype()));
				System.out.println(pt.getGeneration() + " : " + testOnnxName + " : " + 
									pt.getFitness()+ ":" + OnnxHelper.testOnnx(testOnnxName));
			}
			
		}
		
		
		
	}
	
	private static Long generationTrack = (long)1;
	
	private static double accuracyFitness(final Genotype<LayerGene> gt) 
	{
		String onnxname;
		OnnxRegistry registry = OnnxRegistry.getInstance();
		int hashcode = System.identityHashCode(gt);
		
		if(registry.hasEntry(hashcode))
		{
			onnxname = registry.getEntry(hashcode);
		}
		else
		{
			onnxname = OnnxHelper.saveGenotypeAsOnnx(gt);
			registry.addEntry(hashcode, onnxname);
		}
		
		try
		{
			Double accuracyResult = 0.0;
			if(Configs.DATASET.equals("PAMAP2"))
				accuracyResult = OnnxHelper.evaluatePamap2Onnx(onnxname, generationTrack.intValue());
			else
				accuracyResult = OnnxHelper.evaluateOnnx(onnxname, generationTrack.intValue());
			if(accuracyResult!=null)
			{
				 System.out.println(onnxname + ":" + gt.hashCode() +":" + accuracyResult.toString());
				 return accuracyResult;
			}
			else 
			{
				System.err.println("Accuracy Result is null");
				return 0.0;
			}
		}
		catch(Exception e)
		{
			System.err.print(e.getStackTrace());
			return 0.0;
		}
			
	
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
