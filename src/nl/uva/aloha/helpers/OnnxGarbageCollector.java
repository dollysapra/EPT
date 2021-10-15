package nl.uva.aloha.helpers;

import java.io.File;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;

public class OnnxGarbageCollector
{
	private static final OnnxGarbageCollector singletonGCInstance = new OnnxGarbageCollector();
	
	private HashMap<Integer, ArrayList<String>> _filesList;
	//private  _filesToDelete;
	
	
	private OnnxGarbageCollector()
	{
		_filesList = new HashMap<Integer, ArrayList<String>>();
	}
	
	public static OnnxGarbageCollector getInstance()
	{
		return singletonGCInstance;
	}
	
	public void deleteFilesForGeneration(int i)
	{
		if((i<0)||(_filesList.get(i)==null))
			return;
		Iterator<String> it = _filesList.get(i).iterator();
		while(it.hasNext())
		{
			File file = new File(OnnxRegistry._onnxFolder + it.next());
			file.delete();
		}
		_filesList.remove(i);
		
	}
	
	public void removeFileFromDeletionList(int generation, String onnxName)
	{
		if(generation < 0)
			return;
		if ((_filesList.get(generation)!=null) && (_filesList.get(generation).contains(onnxName)) )
			_filesList.get(generation).remove(onnxName);
		
		else if ((_filesList.get(generation+1)!=null) && (_filesList.get(generation+1).contains(onnxName)))
			_filesList.get(generation+1).remove(onnxName);
		
		else if ((_filesList.get(generation -1 )!=null) && (_filesList.get(generation-1).contains(onnxName)))
			_filesList.get(generation-1).remove(onnxName);
	}
	
	public void addFileToListForGeneration(int i, String onnxName)
	{
		if(i<0)
			return;
		if(_filesList.get(i) == null)
			_filesList.put(i, new ArrayList<String>());
		
		
		_filesList.get(i).add(onnxName);
		if(_filesList.get(i-Configs.deleteOnnxAfterGens) !=null)
			deleteFilesForGeneration(i-Configs.deleteOnnxAfterGens);
	}
}
