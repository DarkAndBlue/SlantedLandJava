package slantedland.refactored;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class SlantedLandRefactored {
  static Random rand = new Random();
  static int epochs = 1000;
  static double learning_rate = 0.01;
  
  // real samples
  static Vector[] faces = {
    new Vector(1, 0, 0, 1),
    new Vector(0.9, 0.1, 0.2, 0.8),
    new Vector(0.9, 0.2, 0.1, 0.8),
    new Vector(0.8, 0.1, 0.2, 0.9),
    new Vector(0.8, 0.2, 0.1, 0.9)
  };
  
  public static void main(String[] args) {
    Discriminator discriminator = new Discriminator();
    Generator generator = new Generator();
    
    List<Double> errors_discriminator = new ArrayList<>();
    List<Double> errors_generator = new ArrayList<>();
    
    for (int i = 0; i < epochs; i++) {
      for (Vector face : faces) {
        // Update the discriminator weights from the real face
        discriminator.update_from_image(face);
        
        // Pick a random number to generate a fake face
        double z = rand.nextGaussian();
        
        // Calculate the discriminator error
        double a = discriminator.error_from_image(face);
        Vector b = discriminator.error_from_noise(z).add(a);
        errors_discriminator.add(b.sum());
        
        // Calculate the generator error
        errors_generator.add(generator.error(z, discriminator));
        
        // Build a fake face
        Vector noise = generator.forward(z);
        
        // Update the discriminator weights from the fake face
        discriminator.update_from_noise(noise);
        
        // Update the generator weights from the fake face
        generator.update(z, discriminator);
      }
    }
    
    // Create a window to show error rate and samples
    new Window(generator, errors_discriminator, errors_generator);
  }
  
  static class Discriminator {
    Vector weights;
    double bias;
    
    public Discriminator() {
      weights = Vector.randomized(4);
      bias = rand.nextGaussian();
    }
    
    Vector forward(double x) {
      // Forward pass
      Vector a = weights.dot(x);
      a = a.add(bias);
      return a.sigmoid();
    }
    
    double forward(Vector x) {
      // Forward pass
      return sigmoid(x.dot(weights) + bias);
    }
    private static double sigmoid(double input) {
      return Math.exp(input / (1d + Math.exp(input)));
    }
    
    double error_from_image(Vector image) {
      double prediction = forward(image);
      // We want the prediction to be 1, so the error is -log(prediction)
      return -Math.log(prediction);
    }
    
    Object[] derivatives_from_image(Vector image) {
      double prediction = forward(image);
      
      Vector derivatives_weights = image.inverse().multiply(1 - prediction);
      double derivative_bias = -(1 - prediction);
      return new Object[] { derivatives_weights, derivative_bias };
    }
    
    void update_from_image(Vector x) {
      Object[] ders = derivatives_from_image(x);
      Vector derivatives_weights = (Vector) ders[0];
      double derivative_bias = (double) ders[1];
      
      weights = weights.subtract(derivatives_weights.multiply(learning_rate));
      bias -= learning_rate * derivative_bias;
    }
    
    // We want the prediction to be 0, so the error is -log(1-prediction)
    Vector error_from_noise(double noise) {
      Vector prediction = forward(noise);
      return prediction.subtractReverse(1).log().inverse();
    }
    
    Object[] derivatives_from_noise(Vector noise) {
      double prediction = forward(noise);
      
      Vector derivatives_weights = noise.multiply(prediction);
      double derivative_bias = prediction;
      return new Object[] { derivatives_weights, derivative_bias };
    }
    
    void update_from_noise(Vector noise) {
      Object[] ders = derivatives_from_noise(noise);
      Vector derivatives_weights = (Vector) ders[0];
      double derivative_bias = (double) ders[1];
      
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
    double error(double z, Discriminator discriminator) {
      Vector x = forward(z);
      double y = discriminator.forward(x);
      return -Math.log(y);
    }
    
    Object[] derivatives(double z, Discriminator discriminator) {
      Vector discriminator_weights = discriminator.weights;
      Vector x = forward(z);
      double y = discriminator.forward(x);
      
      double a = -(1 - y);
      Vector b = discriminator_weights.multiply(a);
      b = b.multiply(x);
      Vector factor = b.multiply(x.subtractReverse(1));
      
      Vector derivatives_weights = factor.multiply(z);
      Vector derivative_bias = factor;
      return new Object[] { derivatives_weights, derivative_bias };
    }
    
    void update(double z, Discriminator discriminator) {
      Object[] ders = derivatives(z, discriminator);
      Vector derivatives_weights = (Vector) ders[0];
      Vector derivative_bias = (Vector) ders[1];
      
      weights = weights.subtract(derivatives_weights.multiply(learning_rate));
      biases = biases.subtract(derivative_bias.multiply(learning_rate));
    }
  }
}