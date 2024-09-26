package ann;

import java.util.Random;

//encodes an n-dimensional column vector
//the custom class contains a lot of vector utilities used for this project
//as well as math & some byte parsing
//very extensive
public class Vector {
	//the vector itself is represented as a double array
	double[] vector;
	
	//size based constructor
	public Vector(int n) {
		vector = new double[n];
	}
	
	//value copy constructor
	public Vector(Vector v) {
		vector = new double[v.vector.length];
		for (int i = 0; i < vector.length; i++)
			vector[i] = v.vector[i];
	}

	// byte array parser (used to convert byte array images into floating point representation on which the model trains better)
	public void parseBytes(byte[] arr) {
		// assume arr.length = vector.length
		for (int i = 0; i < arr.length; i++) {
			vector[i] = (arr[i] & 0xFF) / 255.0;
		}
	}

	// random init (used for beginning initialization for the model)
	public void randomInit() {
		Random rand = new Random();
		for (int i = 0; i < vector.length; i++) {
			vector[i] = rand.nextGaussian();
		}
	}
	
	// zero init (used for initializing a zero gradient in the backprop results class)
	public void zeroInit() {
		for (int i = 0; i < vector.length; i++) {
			vector[i] = 0.0;
		}
	}

	// math utility that multiplies a matrix with a vector
	// most of the time spent training is due to the linear algebra operations
	// a very big speed up would come from multithreading these operations
	public static Vector matrixMultiply(Matrix m, Vector v) {
		//check for valid multiply
		if (m.matrix[0].length != v.vector.length)
			return null;

		//create a resultant which holds the final value
		Vector res = new Vector(m.matrix.length);
		//do the multiply
		for (int i = 0; i < res.vector.length; i++) {
			//do the weighted sum of each row multiplied with each component on the input vecotr
			double val = 0.0;
			for (int j = 0; j < v.vector.length; j++) {
				val += v.vector[j] * m.matrix[i][j];
			}
			//place into resultant vector
			res.vector[i] = val;
		}
		//return
		return res;
	}

	// very similar math utility to above method, just for a multiply with the transpose of the provided matrix
	public static Vector matrixMultiplyTransposed(Matrix m, Vector v) {
		//check for valid multiplication
		if (m.matrix.length != v.vector.length)
			return null;
		//create resultant (flipped dimension compared to above method)
		Vector res = new Vector(m.matrix[0].length);
		
		//traverse for each component in the resultant
		for (int i = 0; i < res.vector.length; i++) {
			//do weighted sum across the column of m instead of row
			double val = 0.0;
			for (int j = 0; j < m.matrix.length; j++) {
				// swap order of traversal
				val += v.vector[j] * m.matrix[j][i];
			}
			//place into resultant
			res.vector[i] = val;
		}
		//return
		return res;
	}
	
	// utility that adds a vector object to this vector (component wise)
	// add a vector to this vector
	public void addVector(Vector v) {
		for (int i = 0; i < this.vector.length; i++) {
			this.vector[i] += v.vector[i];
		}
	}
	
	//same as above but subtraction
	public void subtractVector(Vector v) {
		for (int i = 0; i < this.vector.length; i++) {
			this.vector[i] -= v.vector[i];
		}
	}

	// ReLU activation function (required for model to learn)
	private double activate(double x) {
		return (x > 0) ? (x) : (0);
	}
	
	// same as above but returns the derivative at any point of the ReLU activation function
	private double activateDerivative(double x) {
		return (x > 0) ? (1) : (0);
	}

	// a utility that applies activation functions to whole vectors
	public void activate(int layer) {
		//applies softmax activation to the final layer
		if (layer == 1) {
			//calculate the max value for this vector
			double max = max();
			//calculate the sum of all components exponentiated (minus the maximum value for numerical stability)
			double sum = 0.0;
			for (int i = 0; i < vector.length; i++)
				sum += Math.exp(vector[i] - max);
			
			//normalize each exponentiated component by the sum of the exponentiated components
			for (int i = 0; i < vector.length; i++)
				vector[i] = Math.exp(vector[i] - max) / sum;
		} else {
			//use ReLU for hidden layer
			for (int i = 0; i < vector.length; i++)
				vector[i] = activate(vector[i]);
		}
	}

	// same as the above method but applies the derivative of the activation function
	public void activateDerivative(int layer) {
		//only ReLU derivative defintion included since softmax derivative is just the difference anyways (no need to compute)
		if (layer != 2) {
			//activate for each component in the vector
			for (int i = 0; i < vector.length; i++)
				vector[i] = activateDerivative(vector[i]);
		}
	}

	//debugging method to examine the values of vectors
	@Override
	public String toString() {
		String msg = "(";
		for (int i = 0; i < vector.length; i++) {
			msg += String.valueOf(vector[i]);
			if (i != vector.length - 1) {
				msg += ", ";
			} else {
				msg += ")";
			}
		}
		return msg;
	}

	// utility that sums the components
	public double sumComponents() {
		double sum = 0.0;
		for (int i = 0; i < vector.length; i++)
			sum += vector[i];
		return sum;
	}

	// hamard product implementation (element wise product)
	public void hamardProduct(Vector in) {
		for (int i = 0; i < vector.length; i++)
			vector[i] *= in.vector[i];
	}
	
	//static version of above method
	public static Vector hamardProduct(Vector in1, Vector in2) {
		for (int i = 0; i < in1.vector.length; i++)
			in1.vector[i] *= in2.vector[i];
		return in1;
	}
	
	//utility that returns the argument that gives the maximum value in this vector (used for label selection)
	public int argmax() {
		int max = 0;
		for (int j = 1; j < vector.length; j++)
			if (vector[j] > vector[max])
				max = j;
		
		return max;
	}
	
	//utility that returns the maximum component in this vector (used for softmax impl)
	public double max() {
		double max = -Double.MAX_VALUE;
		for (int i = 0; i < vector.length; i++)
			if (vector[i] > max)
				max = vector[i];
		return max;
	}

	// scales and returns a new vector (not state editing)
	public Vector scale(double scale) {
		Vector res = new Vector(vector.length);
		for (int i = 0; i < res.vector.length; i++) {
			res.vector[i] = vector[i] * scale;
		}
		return res;
	}
}