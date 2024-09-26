package ann;

//imports
import java.awt.image.DataBufferByte;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Scanner;

import javax.imageio.ImageIO;


//main class
//runs dataset i/o, training loop, serialization of model, and hyperparameter selection
public class Main {
	//constants relating to training
	//num images is the number of training images in the MNIST dataset (60k)
	public static int NUM_IMAGES;
	//learning rate is hyperparameter that was selected (there is lr decay)
	public static double LEARNING_RATE = 0.05;

	//class constants for the training labels (correct answers) & images themselves (byte arrays)
	static int[] labels;
	static byte[][] images;
	
	//similar constants for test set
	static int[] testLabels;
	static byte[][] testImages;

	//reads the main set of MNIST images (provided the filepaths for the images and labels file)
	//reads based off the encoding described in the MNIST doc (https://yann.lecun.com/exdb/mnist)
	//stores in the static class members
	public static void readMainSet(String imagesFile, String labelsFile) throws Exception {
		//open label file
		File f = new File(labelsFile);
		FileInputStream fin = new FileInputStream(f);
		
		//read
		byte[] data = fin.readAllBytes();
		
		//wrap in buffer for easier reading
		ByteBuffer buff = ByteBuffer.wrap(data);
		
		//remove magic number
		buff.getInt();
		//get the size of the set
		NUM_IMAGES = buff.getInt();
		
		//allocate labels based on set size
		labels = new int[NUM_IMAGES];
		
		//populate labels array
		for (int i = 0; i < NUM_IMAGES; i++) {
			int num = (int) buff.get();
			labels[i] = num;
		}
		
		//close the label file
		fin.close();
		
		//open image file
		f = new File(imagesFile);
		fin = new FileInputStream(f);
		
		//read
		data = fin.readAllBytes();
		
		//wrap in bytebuffer for easier reading
		buff = ByteBuffer.wrap(data);
		
		//remove magic
		buff.getInt();
		//once again get size of set
		NUM_IMAGES = buff.getInt();
		
		//also get bounds on image size (28x28)
		int sizeX = buff.getInt();
		int sizeY = buff.getInt();
		
		//calculate total image size in bytes based on dimensions
		int imageSize = sizeX * sizeY;
		
		//instantiate images array based on the image dimension and the set size
		images = new byte[NUM_IMAGES][imageSize];
		
		//populate images into array
		for (int i = 0; i < NUM_IMAGES; i++) {
			//read each image as byte array
			byte[] image = new byte[imageSize];
			for (int j = 0; j < imageSize; j++) {
				image[j] = buff.get();
			}
			//place into array
			images[i] = image;
		}
		//close file
		fin.close();
	}

	//same as above method but for the testing set
	public static void readTestingSet(String imagesFile, String labelsFile) throws Exception {
		//open label file
		File f = new File(labelsFile);
		FileInputStream fin = new FileInputStream(f);
		
		//read
		byte[] data = fin.readAllBytes();
		
		//wrap in buffer for easier reading
		ByteBuffer buff = ByteBuffer.wrap(data);
		
		//remove magic number
		buff.getInt();
		//get the size of the set
		int numTestElements = buff.getInt();
		
		//allocate labels based on set size
		testLabels = new int[numTestElements];
		
		//populate labels array
		for (int i = 0; i < numTestElements; i++) {
			int num = (int) buff.get();
			testLabels[i] = num;
		}
		
		//close the label file
		fin.close();
		
		//open image file
		f = new File(imagesFile);
		fin = new FileInputStream(f);
		
		//read
		data = fin.readAllBytes();
		
		//wrap in bytebuffer for easier reading
		buff = ByteBuffer.wrap(data);
		
		//remove magic
		buff.getInt();
		//once again get size of set
		numTestElements = buff.getInt();
		
		//also get bounds on image size (28x28)
		int sizeX = buff.getInt();
		int sizeY = buff.getInt();
		
		//calculate total image size in bytes based on dimensions
		int imageSize = sizeX * sizeY;
		
		//instantiate images array based on the image dimension and the set size
		testImages = new byte[numTestElements][imageSize];
		
		//populate images into array
		for (int i = 0; i < numTestElements; i++) {
			//read each image as byte array
			byte[] image = new byte[imageSize];
			for (int j = 0; j < imageSize; j++) {
				image[j] = buff.get();
			}
			//place into array
			testImages[i] = image;
		}
		//close file
		fin.close();
	}
	
	//serialization method for the model
	//this method will save the weights and biases of the trained model in a specific format
	//so it can be read and used later
	public static void saveAnn(Ann ann) throws Exception {
		//create file with .bin ending
		FileOutputStream fos = new FileOutputStream("ann.bin");
		
		//use dataoutputstream class to send floats to ann.bin
		DataOutputStream dos = new DataOutputStream(fos);

		// put in metadata first
		NetworkParameters params = ann.params;
		// first save number of layers
		dos.writeInt(params.layers.length);
		// then the length of each layer
		for (int i = 0; i < params.layers.length; i++) {
			dos.writeInt(params.layers[i].length);
		}

		// then write weight matrix followed by bias vector for each layer
		for (int a = 0; a < ann.weightMatrices.length; a++) {
			//get weights matrix
			Matrix weightMatrix = ann.weightMatrices[a];
			//save each component in weights matrix
			for (int i = 0; i < weightMatrix.matrix.length; i++) {
				for (int j = 0; j < weightMatrix.matrix[i].length; j++) {
					dos.writeDouble(weightMatrix.matrix[i][j]);
				}
			}
			
			//get bias vector
			Vector biasVector = ann.biasVectors[a];
			//save each component in bias vector
			for (int i = 0; i < biasVector.vector.length; i++) {
				dos.writeDouble(biasVector.vector[i]);
			}
		}
		
		//close file
		dos.close();

		fos.close();
	}
	
	//reading from the serialized form of the model
	//will set all weights and biases to the trained ones in the file
	public static Ann readAnn() throws Exception {
		// init file + open it
		File f = new File("ann.bin");
		FileInputStream fin = new FileInputStream(f);
		// read all bytes
		byte[] data = fin.readAllBytes();

		// wrap in bytebuffer to read easier
		ByteBuffer buff = ByteBuffer.wrap(data);

		// read metadata
		// get num layers
		int numLayers = buff.getInt();
		
		//instantiate layer lengths array
		int[] layerLengths = new int[numLayers];
		//traverse and reach each layer's length
		for (int i = 0; i < numLayers; i++) {
			// read each layer's length
			layerLengths[i] = buff.getInt();
			//print to check
			System.out.println(layerLengths[i]);
		}

		// instantiate the ann object using read metadata
		Ann ann = new Ann(new NetworkParameters(layerLengths));

		// update the weight matrices & bias vectors with whats in the file
		for (int l = 1; l < numLayers; l++) {
			//read matrix
			Matrix weightMatrix = ann.weightMatrices[l-1];
			//carefully traverse weights matrix (based on how it was saved)
			for(int i= 0; i< layerLengths[l]; i++) {
				for(int j= 0; j< layerLengths[l-1]; j++) {
					//read each component and set according element in the matrix
					weightMatrix.matrix[i][j] = buff.getDouble();
				}
			}
			
			//read bias vector
			for(int i = 0; i< layerLengths[l]; i++) {
				//read each component and set into array
				ann.biasVectors[l-1].vector[i] = buff.getDouble();
				System.out.println(ann.biasVectors[l-1].vector[i]);
			}
		}
		
		//close input stream to stop any resource leaks
		fin.close();
		
		//finally return ann
		return ann;
	}

	//a helper method that reads a generic file and interprets it as an image file
	//used to test the trained model on out-of-dataset images
	//ex: a handwritten 7 written by myself is predicted to be a 7 at 98% likelihood by the model
	public static byte[] readMyFile(String name) throws Exception {
		//use image io to get a byte representation of the image
		byte[] abgrImage = ((DataBufferByte) ImageIO.read(new File(name)).getRaster().getDataBuffer()).getData();
		
		//transform into grayscale
		byte[] image = new byte[abgrImage.length / 4];
		for (int i = 0; i < image.length; i++) {
			//byte by byte grayscale transform
			byte actual = (byte) ((abgrImage[i * 4 + 1] + abgrImage[i * 4 + 2] + abgrImage[i * 4 + 3]) / 3);
			//place back into array
			image[i] = actual;
		}
		//return
		return image;
	}
	
	//training method for the model
	//this method will prompt the user for the settings of certain hyperparameters
	//like batch size and epoch numbers
	//it will then read the training portion of the MNIST dataset, then train the network
	//the network is already configured to accept the images and output a classification label (0-9)
	//it also tracks the average error of the model (MSE) throughout the epoch as a metric for the user
	public static void trainAnn() throws Exception{
		//user input init
		Scanner user = new Scanner(System.in);
		
		//prompt for num epochs
		System.out.println("Number of training epochs?");
		int numEpochs = user.nextInt();
		
		//prompt for batch size
		System.out.println("Size of training batch?");
		int n = user.nextInt();
		
		//close to prevent resource leak
		user.close();
		
		//read training portion of MNIST dataset
		readMainSet("train-images.idx3-ubyte", "train-labels.idx1-ubyte");

		// create the model based on the parameters
		int[] lengths = { 28 * 28, 20, 10 };
		//use custom classes for the parameters and the model
		NetworkParameters annParameters = new NetworkParameters(lengths);
		Ann ann = new Ann(annParameters);

		// do the training

		// setup the shuffling method (placing indices into a list and shuffling it, then reading in order)
		ArrayList<Integer> samples = new ArrayList<Integer>();
		for (int i = 0; i < NUM_IMAGES; i++)
			samples.add(i);

		//instantiate the custom backpropresults class to hold the gradient accumulated across the entire batch
		BackpropResults average = new BackpropResults(annParameters);

		// init some other useful training parameters
		double scale = (1.0) / ((double) n);
		//indicates the number of training batches underwent in a single training epoch
		//the floor function is used to only do complete batches
		int numMiniBatches = (int) Math.floor(60000.0 / n);

		// training for numEpochs epochs
		for (int epoch = 0; epoch < numEpochs; epoch++) {
			//mark the start time for timing purposes
			long start = System.currentTimeMillis();
			// pre epoch shuffle
			Collections.shuffle(samples);

			// init final error variable (for printing purposes)
			double finalError = 0.0;

			// do each minibatch
			average.zero(); //reset gradient for each batch in the epoch
			//loop through each image
			for (int i = 0; i < n * numMiniBatches; i++) {
				// get a random training image by getting a random index and pulling from it
				int index = samples.get(i);
				
				// vectorize the image
				Vector input = new Vector(28 * 28);
				input.parseBytes(images[index]);

				// do the backpropagation (backprop method also does forward pass)
				BackpropResults singleResults = ann.backprop(input, labels[index]);

				// add this backpropagation's results to the aggregate (scale first)
				for (int j = 0; j < average.biasGradients.length; j++) {
					average.biasGradients[j].addVector(singleResults.biasGradients[j].scale(scale));
					average.weightGradients[j].add(singleResults.weightGradients[j].scale(scale));
				}
				//do the same for the MSE metric
				average.error += singleResults.error * scale;

				// check if the batch is over
				if (i % n == 0) {
					// we do gradient descent based on this batch
					ann.descendGradient(average);
					
					//accumulate the error across the epoch
					finalError += average.error * scale * (1.0 / numMiniBatches);
					
					// zero the gradient prior to the next batch training
					average.zero();
				}
			}
			
			//calculate time taken per epoch
			double time = ((double) (System.currentTimeMillis() - start)) / 1000.0;

			// print out per epoch final error
			System.out.println("Epoch " + (epoch + 1) + ", ANN's MSE error: " + finalError + ". Time Taken (s): " + time);

			// decay the learning rate
			LEARNING_RATE *= 0.65;
		}

		// save this ann to a file
		saveAnn(ann);
		
		//test the ann on the testing set
		testAnn(ann);
	}
	
	//testing method for the model
	//tests it on the testing portion of the MNIST dataset (accuracy)
	public static void testAnn(Ann ann) throws Exception {
		//read the testing set of MNIST
		readTestingSet("t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte");
		//also read the custom file for testing on out-of-dataset identification skills
		byte[] seven = readMyFile("my_7.png");

		// setup the random sampling for the testing set
		ArrayList<Integer>samples = new ArrayList<Integer>();
		for (int i = 0; i < 10000; i++) {
			samples.add(i);
		}
		Collections.shuffle(samples);
		
		//create variable to keep track of each successful identification
		int numSuccessful = 0;
		// do testing over the testing images
		for (int i = 0; i < samples.size(); i++) {
			// get random testing image
			int index = samples.get(i);
			
			// vectorize the image
			Vector input = new Vector(28 * 28);
			input.parseBytes(testImages[index]);

			// feed it to the ann and get the output
			Vector prediction = ann.forward(input)[3];

			// determine the ann's predicted label based on which had the highest likelihood (greedy selection)
			int max = prediction.argmax();

			// check if it was correct
			if (max == testLabels[index]) {
				numSuccessful++;
			}
		}

		// print out success ratio
		double ratio = (double) numSuccessful;
		ratio /= (10000.0);
		ratio *= 100.0;
		System.out.println("\nTesting on 10000 accuracy (%): " + ratio);

		// do final prediction on my (out of dataset) 7
		//vectorize the image
		Vector input = new Vector(28 * 28);
		input.parseBytes(seven);
		
		// feed forward
		Vector prediction = ann.forward(input)[3];
		//print prediction
		System.out.println("\nPrediction for digit (7) outside of dataset: " + prediction.argmax());
	}

	//main function
	//allows the user to choose whether to train a model or read a prexisting one and test that
	public static void main(String[] args) throws Exception {
		//user init
		Scanner user = new Scanner(System.in);
		
		//prompt user
		System.out.println("Train (0) or read (1)?");
		String resp = user.next();
		
		//select based on answer
		if(resp.contains("0")) {
			//train
			trainAnn();
		}else if(resp.contains("1")) {
			//read
			Ann ann = readAnn();
			//test as well
			testAnn(ann);
		}
		
		//close to prevent resource leaks
		user.close();
	}
}