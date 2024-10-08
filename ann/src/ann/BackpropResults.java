package ann;



//simple wrapper class that holds the results of a backpropagation
public class BackpropResults {
	Vector[] biasGradients;
	Matrix[] weightGradients;
	double error;
	
	//zero initialization based on network parameters
	public BackpropResults(NetworkParameters params) {
		weightGradients = new Matrix[params.layers.length - 1];
		biasGradients = new Vector[params.layers.length - 1];

		// start at i = 1 because we ignore input layer (no bias vector for input layer)
		for (int i = 1; i < params.layers.length; i++) {
			int prevLength = params.layers[i - 1].length;
			int length = params.layers[i].length;
			
			//zero init (important that its 0 to ensure no leftover gradient)
			biasGradients[i - 1] = new Vector(length);
			biasGradients[i - 1].zeroInit();
			
			//same for weight matrices
			weightGradients[i - 1] = new Matrix(length, prevLength);
			weightGradients[i - 1].zeroInit();
		}
	}
	
	//zeros the gradient
	public void zero() {
		// make everything set to 0
		for (int i = 0; i < biasGradients.length; i++) {
			biasGradients[i].zeroInit();
		}

		for (int a = 0; a < weightGradients.length; a++) {
			weightGradients[a].zeroInit();
		}

		error = 0.0;
	}
	
	//debug method to view the contents of the back propagation results instance
	@Override
	public String toString() {
		StringBuilder msg = new StringBuilder();
		
		msg.append("Bias Vectors: \n");
		for (int i = 0; i < biasGradients.length; i++)
			msg.append(biasGradients[i].toString() + "\n");
		
		msg.append("Weight Gradients: \n");
		for (int i = 0; i < weightGradients.length; i++)
			msg.append(weightGradients[i].toString() + "\n");
		
		return msg.toString();
	}
}
