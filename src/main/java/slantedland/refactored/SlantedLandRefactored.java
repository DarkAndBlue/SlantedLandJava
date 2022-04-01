package slantedland.refactored;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class SlantedLandRefactored {
  static Random rand = new Random();
  static int epochs = 1000;
  static double learning_rate = 0.01;
  
  static Vector[] faces = {
    new Vector(1, 0, 0, 1),
    new Vector(0.9, 0.1, 0.2, 0.8),
    new Vector(0.9, 0.2, 0.1, 0.8),
    new Vector(0.8, 0.1, 0.2, 0.9),
    new Vector(0.8, 0.2, 0.1, 0.9)
  };
  static double[][][] noise = new double[20][][];
  
  public static void main(String[] args) {
    for (int i = 0; i < 20; i++) {
      noise[i] = new double[2][2];
      for (int j = 0; j < 2; j++) {
        for (int k = 0; k < 2; k++) {
          noise[i][j][k] = rand.nextGaussian();
        }
      }
    }
    
    Discriminator d = new Discriminator();
    Generator g = new Generator();
    
    List<Double> errors_discriminator = new ArrayList<>();
    List<Double> errors_generator = new ArrayList<>();
    
    for (int i = 0; i < epochs; i++) {
      for (Vector face : faces) {
        d.update_from_image(face);
        
        double z = rand.nextGaussian();
        
        double a = d.error_from_image(face);
        Vector b = d.error_from_noise(z);
        Vector c = b.add(a);
        errors_discriminator.add(c.sum());
        
        errors_generator.add(g.error(z, d));
        
        Vector noise = g.forward(z);
        
        d.update_from_noise(noise);
        
        g.update(z, d);
      }
    }
    
    new Window(g, errors_discriminator, errors_generator);
  }
  
  static class Discriminator {
    double[] weights;
    double bias;
    
    public Discriminator() {
      weights = new double[4];
      for (int i = 0; i < weights.length; i++) {
        weights[i] = rand.nextGaussian();
      }
      
      bias = rand.nextGaussian();
    }
    
    Vector forward(double x) {
      Vector a = new Vector(weights).dot(x);
      a = a.add(bias);
      return a.sigmoid();
    }
    
    double forward(Vector x) {
      return Numpy.sigmoid(x.dot(new Vector(weights)) + bias);
    }
    
    double error_from_image(Vector image) {
      double prediction = forward(image);
      return -Math.log(prediction);
    }
    
    Object[] derivatives_from_image(Vector image) {
      double prediction = forward(image);
      double[] derivatives_weights = new double[4];
      for (int i = 0; i < derivatives_weights.length; i++) {
        derivatives_weights[i] = -image.data[i] * (1 - prediction);
      }
      double derivative_bias = -(1 - prediction);
      return new Object[] { derivatives_weights, derivative_bias };
    }
    
    void update_from_image(Vector x) {
      Object[] ders = derivatives_from_image(x);
      double[] derivatives_weights = (double[]) ders[0];
      double derivative_bias = (double) ders[1];
      for (int i = 0; i < weights.length; i++) {
        weights[i] -= learning_rate * derivatives_weights[i];
      }
      bias -= learning_rate * derivative_bias;
    }
    
    Vector error_from_noise(double noise) {
      Vector prediction = forward(noise);
      return prediction.subtractReverse(1).log().inverse();
    }
    
    Object[] derivatives_from_noise(Vector noise) {
      double prediction = forward(noise);
      double[] derivatives_weights = new double[4];
      for (int i = 0; i < derivatives_weights.length; i++) {
        derivatives_weights[i] = noise.data[i] * prediction;
      }
      double derivative_bias = prediction;
      return new Object[] { derivatives_weights, derivative_bias };
    }
    
    void update_from_noise(Vector noise) {
      Object[] ders = derivatives_from_noise(noise);
      double[] derivatives_weights = (double[]) ders[0];
      double derivative_bias = (double) ders[1];
      for (int i = 0; i < weights.length; i++) {
        weights[i] -= learning_rate * derivatives_weights[i];
      }
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
  
    Vector forward(double z) {
      Vector z_weights = weights.multiply(z).add(biases);
      return z_weights.sigmoid();
    }
    
    double error(double z, Discriminator discriminator) {
      Vector x = forward(z);
      double y = discriminator.forward(x);
      return -Math.log(y);
    }
    
    Object[] derivatives(double z, Discriminator discriminator) {
      double[] discriminator_weights = discriminator.weights;
      Vector x = forward(z);
      double y = discriminator.forward(x);
      
      double a = -(1 - y);
      Vector b = new Vector(Numpy.multiply(a, discriminator_weights));
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