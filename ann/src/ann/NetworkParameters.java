package ann;

//simple class to decribe the structure of the model
//uses layer lengths to do so
public class NetworkParameters {
	Layer[] layers;
	
	//initializes the parameters
	public NetworkParameters(int[] layerLengths) {
		// layerLengths array has to be greater than 1 layer
		if (layerLengths.length <= 1) {
			System.err.println("Cannot create network with 1 layer");
			System.exit(0);
		}
		//constructs the layers array
		layers = new Layer[layerLengths.length];
		//iterates and sets each layer length
		for (int i = 0; i < layerLengths.length; i++) {
			Layer layer = new Layer();
			layer.length = layerLengths[i];
			layers[i] = layer;
		}
	}
}