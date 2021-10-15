package nl.uva.aloha.Alterers;

import io.jenetics.AbstractAlterer;
import io.jenetics.AltererResult;
import io.jenetics.Genotype;
import io.jenetics.Phenotype;
import io.jenetics.util.MSeq;
import io.jenetics.util.Seq;
import nl.uva.aloha.genetic.LayerGene;
import nl.uva.aloha.helpers.OnnxHelper;
import nl.uva.aloha.helpers.OnnxRegistry;

public class GenotypeRepeater< T extends Comparable<? super T>>
	extends AbstractAlterer<LayerGene, T>
{

    //private final Factory<Genotype<G>> _factory;

    public GenotypeRepeater() 
    {
    	super(1.0);
    }

	@Override
	public AltererResult<LayerGene, T> alter(Seq<Phenotype<LayerGene, T>> population, long generation) 
	{
		final MSeq<Phenotype<LayerGene, T>> pop = MSeq.of(population);
		OnnxRegistry registry = OnnxRegistry.getInstance();
		
		int alterations =0;
		for (int i = 0, n = pop.size(); i < n; ++i) 
		{
			Phenotype<LayerGene, T> pt = pop.get(i); 
			if (pt.isEvaluated()) 
			{
				alterations++;
				
				String onnxOldGT = registry.getEntry(System.identityHashCode(pt.getGenotype()));
				
				//System.out.println( onnxOldGT + ":" + pop.get(i).getFitness());
				Genotype<LayerGene> gt = Genotype.of(pt.getGenotype().toSeq());
				
				String onnxNewGT = OnnxHelper.saveGenotypeAsOnnx(gt);
				
				registry.addEntry(System.identityHashCode(gt), onnxNewGT);
				ONNXAlteration onnxAlterGt = new ONNXAlteration(OnnxRegistry._onnxFolder+ onnxNewGT);
				
				onnxAlterGt.mutateFrom(OnnxRegistry._onnxFolder+onnxOldGT); 
				onnxAlterGt.updateONNXFile();
				//System.out.println("Repeated from- " + onnxOldGT  +" -To-" + onnxNewGT);
				
				Phenotype<LayerGene, T> ptnew = pt.newInstance(gt,generation);
				pop.set(i, ptnew);
				System.out.println("Repeated from- " + onnxOldGT  +" -To-" + onnxNewGT + ":" + ptnew.isEvaluated());
				
			}
		}
		
		return AltererResult.of(pop.toISeq(),alterations);
	}
}



