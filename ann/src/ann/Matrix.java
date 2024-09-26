package ann;

import java.util.Random;

//encodes a MxN matrix
//like the vector class, it includes utilties for MLP computation
public class Matrix {
	//the actual matrix is encoded as a 2d double array
	double[][] matrix;
	
	//constructor describing both length and width
	public Matrix(int rows, int columns) {
		// random init
		matrix = new double[rows][columns];
	}
	
	//random init (includes possibility for glorot initialization)
	public void randomInit(double skew) {
		Random rand = new Random();
		for (int i = 0; i < matrix.length; i++) {
			for (int j = 0; j < matrix[i].length; j++) {
				matrix[i][j] = rand.nextGaussian() * skew;
			}
		}
	}
	
	// zero init (used for zero gradient initialization in backpropresults class)
	public void zeroInit() {
		for (int i = 0; i < matrix.length; i++) {
			for (int j = 0; j < matrix[i].length; j++) {
				matrix[i][j] = 0.0;
			}
		}
	}
	
	// returns the outer product of two vectors (element wise multiplication of each component of both vectors populating the matrix)
	public static Matrix outerProduct(Vector a, Vector b) {
		//create correct dimension matrix
		Matrix res = new Matrix(a.vector.length, b.vector.length);
		//populate with the multiplies
		for (int i = 0; i < a.vector.length; i++) {
			for (int j = 0; j < b.vector.length; j++) {
				//set each component to the product of the respective elements in each vector
				res.matrix[i][j] = a.vector[i] * b.vector[j];
			}
		}
		//return
		return res;
	}

	// scales and returns a new matrix (not state editing)
	public Matrix scale(double scale) {
		//create new matrix for no state edit
		Matrix res = new Matrix(matrix.length, matrix[0].length);
		//scale each component by the scale factor
		for (int i = 0; i < res.matrix.length; i++) {
			for (int j = 0; j < res.matrix[0].length; j++) {
				res.matrix[i][j] = matrix[i][j] * scale;
			}
		}
		//return
		return res;
	}

	// state editing
	// same as vector (component wise subtraction)
	public void subtract(Matrix m) {
		for (int i = 0; i < this.matrix.length; i++) {
			for (int j = 0; j < this.matrix[i].length; j++) {
				this.matrix[i][j] -= m.matrix[i][j];
			}
		}
	}

	// state editing
	// same as vector (component wise addition)
	public void add(Matrix m) {
		for (int i = 0; i < this.matrix.length; i++) {
			for (int j = 0; j < this.matrix[i].length; j++) {
				this.matrix[i][j] += m.matrix[i][j];
			}
		}
	}

	//debug method to see the values stored in the matrix
	@Override
	public String toString() {
		StringBuilder msg = new StringBuilder();
		for (int i = 0; i < matrix.length; i++) {
			msg.append("(");
			for (int j = 0; j < matrix[i].length; j++) {
				msg.append(matrix[i][j]);
				if (j != matrix[i].length - 1)
					msg.append(", ");
			}
			msg.append(")\n");
		}
		return msg.toString();
	}
}
