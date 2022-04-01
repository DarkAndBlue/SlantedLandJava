package slantedland.refactored;

import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class SlantedLandRefactored {
  static Random rand = new Random();
  static int epochs = 1000;
  static double learning_rate = 0.01;
  
  public static void main(String[] args) {
    Discriminator d = new Discriminator();
    Generator g = new Generator();
    
    List<Double> errors_discriminator = new ArrayList<>();
    List<Double> errors_generator = new ArrayList<>();
    
    for (int i = 0; i < epochs; i++) {
      
      for (double[] face : faces) {
        d.update_from_image(face);
        
        double z = rand.nextGaussian();
        
        double a = d.error_from_image(face);//float64
        double[] b = d.error_from_noise(z);//ndarray
        double[] c = new double[b.length];
        for (int j = 0; j < b.length; j++) {
          c[j] = a + b[j];
        }
        errors_discriminator.add(Numpy.sum(c));
        
        errors_generator.add(g.error(z, d));
        
        double[] noise = g.forward(z);
        
        d.update_from_noise(noise);
        
        g.update(z, d);
      }
    }
    
    new Window(g, errors_discriminator, errors_generator);
  }
  
  static double[][] faces = { { 1, 0, 0, 1 }, { 0.9, 0.1, 0.2, 0.8 }, { 0.9, 0.2, 0.1, 0.8 }, { 0.8, 0.1, 0.2, 0.9 }, { 0.8, 0.2, 0.1, 0.9 } };
  
  static double[][][] noise = new double[20][][];
  
  static {
    for (int i = 0; i < 20; i++) {
      noise[i] = new double[2][2];
      for (int j = 0; j < 2; j++) {
        for (int k = 0; k < 2; k++) {
          noise[i][j][k] = rand.nextGaussian();
        }
      }
    }
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
    
    
    double[] forward(double x) {
      double[] a = Numpy.dot(x, weights);
      for (int i = 0; i < a.length; i++) {
        a[i] += bias;
      }
      return Numpy.sigmoid(a);
    }
    
    double forward(double[] x) {
      return Numpy.sigmoid(Numpy.dot(x, weights) + bias);
    }
    
    double error_from_image(double[] image) {
      double prediction = forward(image);
      return -Math.log(prediction);
    }
    
    Object[] derivatives_from_image(double[] image) {
      double prediction = forward(image);
      double[] derivatives_weights = new double[4];
      for (int i = 0; i < derivatives_weights.length; i++) {
        derivatives_weights[i] = -image[i] * (1 - prediction);
      }
      double derivative_bias = -(1 - prediction);
      return new Object[] { derivatives_weights, derivative_bias };
    }
    
    void update_from_image(double[] x) {
      Object[] ders = derivatives_from_image(x);
      double[] derivatives_weights = (double[]) ders[0];
      double derivative_bias = (double) ders[1];
      for (int i = 0; i < weights.length; i++) {
        weights[i] -= learning_rate * derivatives_weights[i];
      }
      bias -= learning_rate * derivative_bias;
    }
    
    double[] error_from_noise(double noise) {
      double[] prediction = forward(noise);
      return Numpy.inverse(Numpy.log(Numpy.subtractReverse(1, prediction)));  // ndarray
    }
    
    Object[] derivatives_from_noise(double[] noise) {
      double prediction = forward(noise);
      double[] derivatives_weights = new double[4];
      for (int i = 0; i < derivatives_weights.length; i++) {
        derivatives_weights[i] = noise[i] * prediction;
      }
      double derivative_bias = prediction;
      return new Object[] { derivatives_weights, derivative_bias };
    }
    
    void update_from_noise(double[] noise) {
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
    double[] weights;
    double[] biases;
    
    public Generator() {
      weights = new double[4];
      for (int i = 0; i < weights.length; i++) {
        weights[i] = rand.nextGaussian();
      }
      biases = new double[4];
      for (int i = 0; i < biases.length; i++) {
        biases[i] = rand.nextGaussian();
      }
    }
    
    double[] forward(double z) {
      double[] z_weights = new double[4];
      for (int i = 0; i < z_weights.length; i++) {
        z_weights[i] = z * weights[i] + biases[i];
      }
      return Numpy.sigmoid(z_weights);
    }
    
    double error(double z, Discriminator discriminator) {
      double[] x = forward(z);
      double y = discriminator.forward(x);
      return -Math.log(y);
    }
    
    Object[] derivatives(double z, Discriminator discriminator) {
      double[] discriminator_weights = discriminator.weights;
//      double discriminator_bias = discriminator.bias;
      double[] x = forward(z);
      double y = discriminator.forward(x);
      
      double a = -(1 - y);
      double[] b = Numpy.multiply(a, discriminator_weights);
      b = Numpy.multiply(b, x);
      double[] factor = Numpy.multiply(b, Numpy.subtractReverse(1, x));
      
      double[] derivatives_weights = new double[4];
      for (int i = 0; i < derivatives_weights.length; i++) {
        derivatives_weights[i] = factor[i] * z;
      }
      double[] derivative_bias = factor;
      return new Object[] { derivatives_weights, derivative_bias };
    }
    
    void update(double z, Discriminator discriminator) {
//      double error_before = error(z, discriminator);
      Object[] ders = derivatives(z, discriminator);
      double[] derivatives_weights = (double[]) ders[0];
      double[] derivative_bias = (double[]) ders[1];
      for (int i = 0; i < weights.length; i++) {
        weights[i] -= learning_rate * derivatives_weights[i];
      }
      for (int i = 0; i < weights.length; i++) {
        biases[i] -= learning_rate * derivative_bias[i];
      }
//      double error_after = error(z, discriminator);
    }
  }
}