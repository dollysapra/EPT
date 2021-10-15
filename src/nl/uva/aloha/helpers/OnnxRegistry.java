package nl.uva.aloha.helpers;

import java.util.Date;
import java.util.HashMap;

public class OnnxRegistry 
{
	private static final OnnxRegistry singletonRegistryInstance = new OnnxRegistry();;
	private static HashMap<Integer, String> Register;
	
	
	public static String _onnxFolder = Configs.OnnxWorkspacePath +  new Date().getTime() + "/";
	
	private OnnxRegistry()
	{
		Register = new HashMap<Integer, String>();
	}
	
	public static OnnxRegistry getInstance()
	{
		return singletonRegistryInstance;
	}
	
	public void addEntry(int GenotypeUniquecode,String OnnxName)
	{
		if(hasEntry(new Integer(GenotypeUniquecode)))
			System.err.println("Duplicate Entry?");
		Register.put(new Integer(GenotypeUniquecode), OnnxName);
	}
	
	public Boolean hasEntry(int GenotypeUniquecode)
	{
		return Register.containsKey(new Integer(GenotypeUniquecode));
	}
	
	public String getEntry(int GenotypeUniquecode)
	{
		return Register.get(new Integer(GenotypeUniquecode));
	}
}
