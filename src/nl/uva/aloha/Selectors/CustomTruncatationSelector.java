package nl.uva.aloha.Selectors;

import io.jenetics.Gene;
import io.jenetics.Optimize;
import io.jenetics.Phenotype;
import io.jenetics.Selector;
import io.jenetics.util.ISeq;
import io.jenetics.util.MSeq;
import io.jenetics.util.RandomRegistry;
import io.jenetics.util.Seq;

public class CustomTruncatationSelector < G extends Gene<?, G>, C extends Comparable<? super C> >
											implements Selector<G, C>
{
	private final int _n;
	//private final int _copySelectorOffset;
	
	public CustomTruncatationSelector(final int n ) 
	{
		if (n < 1) {
			throw new IllegalArgumentException(String.format("n must be greater or equal 1, but was %d.", n));
		}
		_n = n;
		
		//_copySelectorOffset = selectorOffsetForCopy;
	}
	
	public CustomTruncatationSelector() 
	{
		this(Integer.MAX_VALUE);
	}

	@Override
	public ISeq<Phenotype<G, C>> select(Seq<Phenotype<G, C>> population, int count, Optimize opt)
	{
		final MSeq<Phenotype<G, C>> selection = MSeq.ofLength(population.isEmpty() ? 0 : count);

			if (count > 0 && !population.isEmpty()) 
			{
				final MSeq<Phenotype<G, C>> copy = population.asISeq().copy();
				copy.sort((a, b) -> opt.<C>descending().compare(a.getFitness(), b.getFitness()));

				int size = count;
				//int copyOffset =0;
				do 
				{
					
					final int length = Math.min(Math.min(copy.size(), size), _n);
					for (int i = 0; i < length; ++i) 
					{
						if(size>=_n)
							selection.set((count - size) + i, copy.get(i));
						else
							selection.set((count - size) + i, copy.get(RandomRegistry.getRandom().nextInt(copy.size())));
						
					}

					size -= length;
					//copyOffset += _copySelectorOffset;
					
				} while (size > 0);
			}

			return selection.toISeq();
	}

	@Override
	public boolean equals(final Object obj) 
	{
		return obj == this || obj != null && getClass() == obj.getClass();
	}

	@Override
	public String toString() 
	{
		return getClass().getName();
	}

	
}
