package nl.uva.aloha.Alterers;

import java.util.Random;

import io.jenetics.Chromosome;
import io.jenetics.Genotype;
import io.jenetics.Mutator;
import io.jenetics.MutatorResult;
import io.jenetics.Phenotype;
import io.jenetics.util.MSeq;
import nl.uva.aloha.genetic.LayerGene;
import nl.uva.aloha.helpers.OnnxHelper;
import nl.uva.aloha.helpers.OnnxRegistry;

public class LayerGeneMutator<T extends Comparable<? super T>>
		extends Mutator<LayerGene,T>
{
	private double _rate;
	
	public LayerGeneMutator(double probability, double rate) 
	{
		super(probability);
		_rate = rate;
	}
	
	@Override
	protected MutatorResult<Phenotype<LayerGene, T>> mutate(final Phenotype<LayerGene, T> phenotype,
																 final long generation,final double p,	final Random random) 
	{
		if(phenotype.isEvaluated())
			return super.mutate(phenotype, generation, p, random);
		else return(MutatorResult.of(phenotype));
	}
	
	@Override
	protected MutatorResult<Genotype<LayerGene>> mutate(
			final Genotype<LayerGene> genotype,
			final double p,
			final Random random) 
	{
		
		
		try 
		{
			OnnxRegistry registry = OnnxRegistry.getInstance();
			String onnxOldGT = registry.getEntry(System.identityHashCode(genotype));
			
			
			//Choosing the Chromosome index for mutation. - To reduce number of chromosomes (here only 1 chromosome is mutated) 
			final int chIndex = random.nextInt((genotype.length()-4) + 1); //Avoid Last 3 and the first one = total 4 chromosomes
			MSeq<Chromosome<LayerGene>> g1 = genotype.toSeq().copy();
			g1.set(chIndex, super.mutate(genotype.get(chIndex), p, random).getResult());
			MutatorResult<Genotype<LayerGene>> mutatorResult = MutatorResult.of(Genotype.of(g1));
			
			//Choosing chromosomes the standard way
			//MutatorResult<Genotype<LayerGene>> mutatorResult = super.mutate(genotype, p, random);
			
			
			String onnxNewGT = OnnxHelper.saveGenotypeAsOnnx(mutatorResult.getResult());
			
			registry.addEntry(System.identityHashCode(mutatorResult.getResult()), onnxNewGT);
			ONNXAlteration onnxAlterGt = new ONNXAlteration(OnnxRegistry._onnxFolder+ onnxNewGT);
			
			onnxAlterGt.mutateFrom(OnnxRegistry._onnxFolder+onnxOldGT); 
			onnxAlterGt.updateONNXFile();
			
			System.out.println("Mutated from- " + onnxOldGT  +" -To-" + onnxNewGT);

			return mutatorResult;
		}
		catch(Exception e)
		{
			System.err.println(e.getMessage());
			return MutatorResult.of(genotype);
		}
		
	}
	
	
	@Override
	protected LayerGene mutate(final LayerGene gene, final Random random) 
	{
		return mutate(gene);
	}
	

	private LayerGene mutate(final LayerGene gene) 
	{
		return gene.mutate(_rate);
	}
	
	
}
