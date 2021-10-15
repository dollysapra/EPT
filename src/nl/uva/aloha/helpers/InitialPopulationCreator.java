package nl.uva.aloha.helpers;

import java.util.List;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ForkJoinPool;
import java.util.stream.Collectors;

import io.jenetics.Genotype;
import nl.uva.aloha.genetic.DNNCodec;
import nl.uva.aloha.genetic.LayerGene;

public class InitialPopulationCreator 
{
	private List<Genotype<LayerGene>> genotypes;
	private int _populationSize;

	public InitialPopulationCreator(int populationSize)
	{
		
		_populationSize = populationSize;
	}
	
	public List<Genotype<LayerGene>> createPopulation()
	{
		Genotype<LayerGene> genotypeEncoding = DNNCodec.ENCODING;
		int count = 0; 
		
		 final int parallelism = 3;
		 
	        ForkJoinPool forkJoinPool = null;

	        try {
	            forkJoinPool = new ForkJoinPool(parallelism);
	            forkJoinPool.submit(() ->

	                    //parallel stream invoked here
			            genotypes = genotypeEncoding.instances().limit(new Double(_populationSize*1.1).longValue()).unordered().parallel().
						filter(g->{ String name = OnnxHelper.saveGenotypeAsOnnx(g); 
									Boolean predicateresult = (OnnxHelper.evaluateOnnx(name, 0)>0.11);
									System.out.println(name);
									if(predicateresult)
										OnnxRegistry.getInstance().addEntry(System.identityHashCode(g), name);
									/*Useful if a lot of networks are not learning - however - modifying learning rate is more useful 
									 *than resetting weights and checking again! 
									 *  if(predicateresult == false) //Try again with different initialization.
										{
											String newname = OnnxHelper.saveGenotypeAsOnnx(g); 
											predicateresult = (OnnxHelper.evaluateOnnx(newname, 0)>0.11);
											System.out.println(name + "(P) - (N)" + newname);
										}*/
									
									return  predicateresult;
									}).
									limit(_populationSize).collect(Collectors.<Genotype<LayerGene>>toList())
	            
	           ).get(); //this makes it an overall blocking call

	        } catch (InterruptedException | ExecutionException e) {
	            e.printStackTrace();
	        } finally {
	            if (forkJoinPool != null) {
	                forkJoinPool.shutdown(); //always remember to shutdown the pool
	            }
	        }

	        
	        

		
		
		//TEST WHAT happens if weights are reset!
		/*System.out.println("RESET");
		genotypes.parallelStream().forEach(gtp->{ String name = OnnxHelper.saveGenotypeAsOnnx(gtp); 
													OnnxHelper.evaluateOnnx(name, 0);
													System.out.println(name);
												});*/
		
		return genotypes;
		
	}
	
	public static void main(final String[] args)
	{
		List<Genotype<LayerGene>> genotypes = new InitialPopulationCreator(15).createPopulation();
	}
}
