package backend;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;

import weka.classifiers.lazy.IBk;
import weka.core.Instance;
import weka.core.Instances;

public class WekaClustering {
	
	public WekaClustering(String pathToArffFile){
		try {
			clusterArffData(pathToArffFile);
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

	private BufferedReader readDataFile(String filename) {
		BufferedReader inputReader = null;
		try {
			inputReader = new BufferedReader(new FileReader(filename));
		} catch (FileNotFoundException ex) {
			System.err.println("File not found: " + filename);
		}
		return inputReader;
	}

	private void clusterArffData(String pathToArffFile) throws Exception {
		BufferedReader datafile = readDataFile(pathToArffFile);
		Instances data = new Instances(datafile);
		data.setClassIndex(data.numAttributes() - 1);
		// do not use first and second
		Instance first = data.instance(0);
		Instance second = data.instance(1);
		data.delete(0);
		data.delete(1);
		IBk ibk = new IBk();
		ibk.buildClassifier(data);
		System.out.println(ibk.getNumTraining());
		double class1 = ibk.classifyInstance(first);
		double class2 = ibk.classifyInstance(second);
		System.out.println("first: " + class1 + "\nsecond: " + class2);
	}

}
