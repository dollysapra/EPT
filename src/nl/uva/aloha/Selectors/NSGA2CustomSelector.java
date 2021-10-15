package nl.uva.aloha.Selectors;

import java.util.Comparator;
import java.util.function.ToIntFunction;

import io.jenetics.Gene;
import io.jenetics.Optimize;
import io.jenetics.Phenotype;
import io.jenetics.ext.moea.ElementComparator;
import io.jenetics.ext.moea.ElementDistance;
import io.jenetics.ext.moea.NSGA2Selector;
import io.jenetics.ext.moea.Vec;
import io.jenetics.util.ISeq;
import io.jenetics.util.MSeq;
import io.jenetics.util.RandomRegistry;
import io.jenetics.util.Seq;

public class NSGA2CustomSelector < G extends Gene<?,G>,     C extends Comparable<? super C>     > 
						extends  NSGA2Selector<G, C>
{
	
	private int _randomSurvivorCount = 0;

	public NSGA2CustomSelector(Comparator<? super C> dominance, ElementComparator<? super C> comparator,
			ElementDistance<? super C> distance, ToIntFunction<? super C> dimension, int randomSurvivorCount) {
		super(dominance, comparator, distance, dimension);
		
		_randomSurvivorCount = randomSurvivorCount;
	}
	
@SuppressWarnings("unchecked")
@Override
public ISeq<Phenotype<G, C>> select(Seq<Phenotype<G, C>> population, int count, Optimize opt) {
	
	final MSeq<Phenotype<G, C>> selection = MSeq.ofLength(population.isEmpty() ? 0 : count);
	int bests = count - _randomSurvivorCount;
			
	selection.setAll(  super.select(population, bests, opt).asMSeq().copy());
	
	
	for(int i=0; i< _randomSurvivorCount; i++)
	{
		selection.set( bests + i, (population.asISeq().copy().get(RandomRegistry.getRandom().nextInt(population.size()))));
	}
	return selection.toISeq();
}
	
	
	public static <G extends Gene<?, G>, T, V extends Vec<T>>
	NSGA2CustomSelector<G, V> ofVec(int randomSurvivorCount) {
		return new NSGA2CustomSelector<>(
			Vec::dominance,
			Vec::compare,
			Vec::distance,
			Vec::length,
			randomSurvivorCount
		);
	}
	
}
