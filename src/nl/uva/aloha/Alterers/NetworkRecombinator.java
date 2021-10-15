package nl.uva.aloha.Alterers;

import static io.jenetics.internal.math.random.indexes;

import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.stream.Collectors;

import nl.uva.aloha.GAMain;
import io.jenetics.AbstractAlterer;
import io.jenetics.AltererResult;
import io.jenetics.Chromosome;
import io.jenetics.Genotype;
import io.jenetics.Phenotype;
import io.jenetics.util.MSeq;
import io.jenetics.util.RandomRegistry;
import io.jenetics.util.Seq;
import nl.uva.aloha.genetic.DualLayerChromosome;
import nl.uva.aloha.genetic.GenotypeTranslation;
import nl.uva.aloha.genetic.LayerGene;
import nl.uva.aloha.genetic.SkipCnxnDualLayerChromosome;
import nl.uva.aloha.helpers.OnnxHelper;
import nl.uva.aloha.helpers.OnnxRegistry;


public class NetworkRecombinator<T extends Comparable<? super T>>
			extends AbstractAlterer<LayerGene, T>

{
	
	private  int _order = 2;
	
	public NetworkRecombinator(double probability) 
	{
		super(probability);
		
	}

	protected NetworkRecombinator(double probability, int order) 
	{
		super(probability);
		if(order >= 2)
			_order = order;
	}
	
	public int getOrder() {
		return _order;
	}
	
	
	@Override
	public final AltererResult<LayerGene, T> alter(
		final Seq<Phenotype<LayerGene, T>> population,
		final long generation
	) {
		final AltererResult<LayerGene, T> result;
		if (population.size() >= 2) {
			final Random random = RandomRegistry.getRandom();
			final int order = Math.min(_order, population.size());

			// Selection of the individuals for recombination.
			//final IntFunction<int[]> individuals = i -> {
		//		final int[] ind = subset(population.size(), order, random);
		//		ind[0] = i;
		//		return ind;
		//	};


			final MSeq<Phenotype<LayerGene, T>> pop = MSeq.of(population);
			int count = 0;
			
			//Exponential decay to probability
			//List<Integer> indexList = indexes(random, population.size(), _probability * Math.pow(0.95,generation)).boxed().collect(Collectors.toList());
			
			//Adaptive probability
			//List<Integer> indexList = indexes(random, population.size(), _probability * GAMain.rateOfChange).boxed().collect(Collectors.toList());
			
			//Fixed probability for every tim
			List<Integer> indexList = indexes(random, population.size(), _probability).boxed().collect(Collectors.toList());
			
			Collections.shuffle(indexList);
			int[] indexes = indexList.stream().mapToInt(i->i.intValue()).toArray();
					
			
			for(int i=0; i<indexes.length/2; i++)
			{
				count+= recombine(pop, new int[]{indexes[2*i], indexes[2*i+1]}, generation );
			}

			result = AltererResult.of(pop.toISeq(), count);
		} else {
			result = AltererResult.of(population.asISeq());
		}

		return result;
	}
	

	protected int recombine(final MSeq<Phenotype<LayerGene, T>> population,
							final int[] individuals,
							final long generation) 
	{
		
		assert individuals.length == 2 : "Required order of 2";
		final Random random = RandomRegistry.getRandom();

		final Phenotype<LayerGene, T> pt1 = population.get(individuals[0]);
		final Phenotype<LayerGene, T> pt2 = population.get(individuals[1]);
		final Genotype<LayerGene> gt1 = pt1.getGenotype();
		final Genotype<LayerGene> gt2 = pt2.getGenotype();

		//Choosing the Chromosome index for crossover.
		final int chIndex = random.nextInt(Math.min(gt1.length()-4, gt2.length()-4) + 1); //Avoid Last 3 and the first one = total 4 chromosomes
		
		final MSeq<Chromosome<LayerGene>> c1 = gt1.toSeq().copy();
		final MSeq<Chromosome<LayerGene>> c2 = gt2.toSeq().copy();
		
		final MSeq<LayerGene> genes1 = c1.get(chIndex).toSeq().copy();
		final MSeq<LayerGene> genes2 = c2.get(chIndex).toSeq().copy();
		
		try
		{
			//int index = RandomRegistry.getRandom().nextInt(Math.min(genes1.length(), genes2.length()));
			
			if(c1.get(chIndex) instanceof SkipCnxnDualLayerChromosome)
			{
				crossoverskip(genes1, genes2); 
				
				c1.set(chIndex, c1.get(chIndex).newInstance(genes2.toISeq()));
				c2.set(chIndex, c2.get(chIndex).newInstance(genes1.toISeq()));
				
			}
			else
				c1.swap(chIndex, c2); 	//Swap only the chosen chromosome
			
			/**
			 * Swap one gene in selected chromosome
			 */
			//c1.set(chIndex, c1.get(chIndex).newInstance(genes1.toISeq()));
			//c2.set(chIndex, c2.get(chIndex).newInstance(genes2.toISeq()));
			
			
			/**
			 * Swap rest of genotype 
			 */
			//for(int swapi = chIndex+1; swapi<gt1.length(); swapi++)
			//	c1.swap(swapi, c2);
			//Check
			
			
			
			
			//ONNXWORK TODO: Clean code and make a separate function
			Genotype<LayerGene> newGT1 = Genotype.of(c1);
			Genotype<LayerGene> newGT2 = Genotype.of(c2);
			String onnx_gt1 = OnnxHelper.saveGenotypeAsOnnx(newGT1);
			String onnx_gt2 = OnnxHelper.saveGenotypeAsOnnx(newGT2);
			
			OnnxRegistry registry = OnnxRegistry.getInstance();
			registry.addEntry(System.identityHashCode(newGT1), onnx_gt1);
			registry.addEntry(System.identityHashCode(newGT2), onnx_gt2);
			
			ONNXAlteration onnxAlter_gt1 = new ONNXAlteration(OnnxRegistry._onnxFolder+ onnx_gt1);
			ONNXAlteration onnxAlter_gt2 = new ONNXAlteration(OnnxRegistry._onnxFolder+ onnx_gt2);
			
			onnxAlter_gt1.recombineFrom(OnnxRegistry._onnxFolder + registry.getEntry(System.identityHashCode(gt1)), 
										OnnxRegistry._onnxFolder + registry.getEntry(System.identityHashCode(gt2)) );
			
			onnxAlter_gt2.recombineFrom(OnnxRegistry._onnxFolder + registry.getEntry(System.identityHashCode(gt1)), 
										OnnxRegistry._onnxFolder + registry.getEntry(System.identityHashCode(gt2)) );
			
			onnxAlter_gt1.updateONNXFile();
			onnxAlter_gt2.updateONNXFile();
			
			System.out.println(generation + "-Recombined from- " + registry.getEntry(System.identityHashCode(gt1)) +
								":" + registry.getEntry(System.identityHashCode(gt2)) +
								" To-" + onnx_gt1 + ":" +onnx_gt2);
			System.out.println(GenotypeTranslation.getSignature(gt1) + ":" + GenotypeTranslation.getSignature(gt2) +
								" To- " + GenotypeTranslation.getSignature(newGT1) + ":" + GenotypeTranslation.getSignature(newGT2));
			//Creating two new Phenotypes and exchanging it with the old.
			population.set(
				individuals[0],
				pt1.newInstance(newGT1, generation)
			);
			population.set(
				individuals[1],
				pt2.newInstance(newGT2, generation)
			);
		}
		catch(Exception e)
		{
			System.err.println(e.getMessage());
		}
				
		return getOrder();
	}

	
	private void crossover( final MSeq<LayerGene> that, final MSeq<LayerGene> other, final int index) 
	{
		assert index >= 0 : String.format("Crossover index must be within [0, %d) but was %d", that.length(), index );
		that.swap(index, Math.min(that.length(), other.length()), other, index);
	}
	
	private void crossoverskip( final MSeq<LayerGene> that, final MSeq<LayerGene> other) 
	{
		that.swap(that.length()- 2,that.length(), other, other.length() - 2); // len - 2 because start is inclusive and end is exclusive for index
	}
	
}
