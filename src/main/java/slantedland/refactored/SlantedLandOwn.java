package slantedland.refactored;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class SlantedLandOwn {
  static Random rand = new Random();
  static int epochs = 1000;
  static double learning_rate = 0.01;
  
  // real samples
  static Vector[] targets = {
    new Vector(1, 0, 0, 1),
    new Vector(0.9, 0.1, 0.2, 0.8),
    new Vector(0.9, 0.2, 0.1, 0.8),
    new Vector(0.8, 0.1, 0.2, 0.9),
    new Vector(0.8, 0.2, 0.1, 0.9)
  };
  
  public static void main(String[] args) {
    Discriminator discriminator = new Discriminator();
    Generator generator = new Generator();
    
    List<Double> discriminatorErrors = new ArrayList<>();
    List<Double> generatorErrors = new ArrayList<>();
    
    for (int i = 0; i < epochs; i++) {
      for (Vector target : targets) {
        // train the discriminator on real images
        discriminator.trainOnTarget(target);
        
        // Pick a random number to generate a fake face
        double noiseInput = rand.nextGaussian();
        
        // Calculate the discriminator error
        double targetPrediction = discriminator.predictionTarget(target);
        Vector combined = discriminator.predictionNoise(noiseInput).add(targetPrediction);
        discriminatorErrors.add(combined.sum());
        
        // Calculate the generator error
        generatorErrors.add(generator.calculateError(noiseInput, discriminator));
        
        // Build a fake face
        Vector generated = generator.forward(noiseInput);
        
        // Update the discriminator weights from the fake face
        discriminator.trainOnFake(generated);
        
        // Update the generator weights from the fake face
        generator.train(noiseInput, discriminator);
      }
    }
    
    // Create a window to show error rate and samples
    new Window(generator, discriminatorErrors, generatorErrors);
  }
  
  static class Discriminator {
    Vector weights;
    double bias;
    
    public Discriminator() {
      weights = Vector.randomized(4);
      bias = rand.nextGaussian();
    }
    
    Vector forwardNoise(double noise) {
      // Forward pass
      Vector a = weights.dot(noise);
      a = a.add(bias);
      return a.sigmoid();
    }
    
    double forward(Vector x) {
      // Forward pass
      return sigmoid(x.dot(weights) + bias);
    }
    
    double predictionTarget(Vector target) {
      double prediction = forward(target);
      // We want the prediction to be 1, so the error is -log(prediction)
      return -Math.log(prediction);
    }
    
    void trainOnTarget(Vector targets) {
      double prediction = forward(targets);
  
      Vector derivatives_weights = targets.inverse().multiply(1 - prediction);
      double derivative_bias = -(1 - prediction);
      
      weights = weights.subtract(derivatives_weights.multiply(learning_rate));
      bias -= learning_rate * derivative_bias;
    }
    
    // We want the prediction to be 0, so the error is -log(1-prediction)
    Vector predictionNoise(double noise) {
      Vector prediction = forwardNoise(noise);
      return prediction.subtractReverse(1).log().inverse();
    }
    
    void trainOnFake(Vector generated) {
      // prediction on how real the generated image is
      double prediction = forward(generated);
  
      Vector derivatives_weights = generated.multiply(prediction);
      double derivative_bias = prediction;
      
      weights = weights.subtract(derivatives_weights.multiply(learning_rate));
      bias -= learning_rate * derivative_bias;
    }
  }
  
  static class Generator {
    Vector weights;
    Vector biases;
    
    public Generator() {
      weights = Vector.randomized(4);
      biases = Vector.randomized(4);
    }
    
    // Forward pass
    Vector forward(double z) {
      Vector z_weights = weights.multiply(z).add(biases);
      return z_weights.sigmoid();
    }
    
    // We want the prediction to be 0, so the error is -log(1-prediction)
    double calculateError(double z, Discriminator discriminator) {
      Vector x = forward(z);
      double y = discriminator.forward(x);
      return -Math.log(y);
    }
    
    void train(double noiseInput, Discriminator discriminator) {
      Vector discriminatorWeights = discriminator.weights;
      Vector generatorPrediction = forward(noiseInput);
      double discriminatorPrediction = discriminator.forward(generatorPrediction);
  
      double a = -(1 - discriminatorPrediction);
      Vector b = discriminatorWeights.multiply(a);
      b = b.multiply(generatorPrediction);
      Vector factor = b.multiply(generatorPrediction.subtractReverse(1));
  
      Vector derivatives_weights = factor.multiply(noiseInput);
      Vector derivative_bias = factor;
      
      weights = weights.subtract(derivatives_weights.multiply(learning_rate));
      biases = biases.subtract(derivative_bias.multiply(learning_rate));
    }
  }
  
  private static double sigmoid(double input) {
    return Math.exp(input / (1d + Math.exp(input)));
  }
}