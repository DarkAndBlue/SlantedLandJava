package mnistgan;

import slantedland.refactored.Vector;

import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.ThreadLocalRandom;

public class MNISTGAN extends JFrame {
  public static void main(String[] args) {
    new MNISTGAN();
  }
  
  public MNISTGAN() {
//    setTitle("MNIST GENERATION");
    setDefaultCloseOperation(EXIT_ON_CLOSE);
    
    setSize(1800, 600);
    
    Scene scene = new Scene();
    scene.setSize(1600, 800);
    add(scene);
    
    setVisible(true);
    setLocationRelativeTo(null);
    
    while (true) {
      scene.repaint();
    }
  }
  
  class Scene extends JPanel {
    static Random random = new Random();
    static int imageSize = 16;
    static double learning_rate = 0.03;
    static int targetsPerEpoch = 5;
    Discriminator discriminator;
    Generator generator;
    List<Double> discriminatorErrors = new ArrayList<>();
    List<Double> generatorErrors = new ArrayList<>();
    
    public Scene() {
      discriminator = new Discriminator();
      generator = new Generator();
      
      setDoubleBuffered(true);
    }
    
    BufferedImage drawingImage = new BufferedImage(imageSize, imageSize, BufferedImage.TYPE_INT_RGB);
    int maxErrorSampleSize = 3000;
    
    public void paint(Graphics graphics) {
      super.paint(graphics);
      updateNeuralNetworks();
      
      double input = Math.random();
      Vector forward = generator.forward(input);
      
      for (int x = 0; x < imageSize; x++) {
        for (int y = 0; y < imageSize; y++) {
          int i = x + y * imageSize;
          int brightness = (int) (255d - forward.data[i] * 255d);
          drawingImage.setRGB(x, y, new Color(brightness, brightness, brightness).getRGB());
        }
      }
      graphics.drawImage(drawingImage, 0, 0, getWidth() / 3, getHeight(), null);
      
      graphics.setColor(Color.blue);
      drawList(graphics, generatorErrors, getWidth() / 3, 0.05);
      graphics.setColor(Color.red);
      drawList(graphics, discriminatorErrors, getWidth() / 3 * 2, 0.01);
      
      while (generatorErrors.size() > maxErrorSampleSize)
        generatorErrors.remove(0);
      
      while (discriminatorErrors.size() > maxErrorSampleSize)
        discriminatorErrors.remove(0);
    }
    
    void drawList(Graphics graphics, List<Double> list, int xOffset, double yScale) {
      double sizeX = getWidth() / 3d / maxErrorSampleSize;
      int yOffset = avgY(list);
      
      int[] x = new int[list.size()];
      int[] y = new int[list.size()];
      
      for (int i = 0; i < list.size(); i++) {
        double value = list.get(i);
        
        x[i] = (int) (i * sizeX) + xOffset;
        y[i] = (int) ((value - yOffset) / yScale) + getHeight() / 2;
      }
      
      graphics.drawPolyline(x, y, x.length);
    }
    
    int avgY(List<Double> list) {
      double sum = 0;
      for (int i = 0; i < list.size(); i++) {
        sum += list.get(i);
      }
      return (int) (sum / list.size());
    }
    
    void updateNeuralNetworks() {
      for (int j = 0; j < targetsPerEpoch; j++) {
        Vector target = generateTarget();
        // train the discriminator on real images
        discriminator.trainOnTarget(target);
        
        // Pick a random number to generate a fake face
        double noise = random.nextGaussian();
        
        // Calculate the discriminator error
        double targetPrediction = discriminator.calculateError(target, noise, generator);
        discriminatorErrors.add(targetPrediction);
        
        // Calculate the generator error
        generatorErrors.add(generator.calculateError(noise, discriminator));
        
        // Build a fake face
        Vector generated = generator.forward(noise);
        
        // Update the discriminator weights from the fake face
        discriminator.trainOnFake(generated);
        
        // Update the generator weights from the fake face
        generator.train(noise, discriminator);
      }
    }
    
    // real samples
    static Vector generateTarget() {
      BufferedImage image = new BufferedImage(imageSize, imageSize, BufferedImage.TYPE_INT_BGR);
      Graphics2D graphics = image.createGraphics();
      graphics.setColor(Color.black);
      graphics.fillRect(0, 0, imageSize, imageSize);
      graphics.setColor(Color.white);
      graphics.drawLine(random(), random(), imageSize + random(), imageSize + random());
      
      double[] values = new double[imageSize * imageSize];
      for (int j = 0; j < imageSize * imageSize; j++) {
        int color = image.getRGB(j % imageSize, j / imageSize);
        values[j] = new Color(color).getRed() / 255d;
      }
      return new Vector(values);
    }
    
    static int random() {
      return ThreadLocalRandom.current().nextInt(-2, 2);
    }
    
    static class Discriminator {
      Vector weights;
      double bias;
      
      public Discriminator() {
        weights = Vector.randomized(imageSize * imageSize);
        bias = random.nextGaussian();
      }
  
      // Forward pass
      double forward(Vector x) {
        return sigmoid(x.dot(weights) + bias);
      }
      
      void trainOnTarget(Vector targets) {
        double prediction = forward(targets);
        
        Vector derivatives_weights = targets.inverse().multiply(1 - prediction);
        double derivative_bias = -(1 - prediction);
        
        weights = weights.subtract(derivatives_weights.multiply(learning_rate));
        bias -= learning_rate * derivative_bias;
      }
      
      void trainOnFake(Vector generated) {
        // prediction on how real the generated image is
        double prediction = forward(generated);
        
        Vector derivatives_weights = generated.multiply(prediction);
        double derivative_bias = prediction;
        
        weights = weights.subtract(derivatives_weights.multiply(learning_rate));
        bias -= learning_rate * derivative_bias;
      }
      
      public double calculateError(Vector target, double noise, Generator generator) {
        Vector generated = generator.forward(noise);
        double predictionFake = forward(generated);
        double predictionReal = 1 - forward(target);
        
        return predictionReal + predictionFake;
      }
    }
    
    static class Generator {
      Vector weights;
      Vector biases;
      
      public Generator() {
        weights = Vector.randomized(imageSize * imageSize);
        biases = Vector.randomized(imageSize * imageSize);
      }
      
      // Forward pass
      Vector forward(double noise) {
        Vector z_weights = weights.multiply(noise).add(biases);
        return z_weights.sigmoid();
      }
      
      // We want the prediction to be 0, so the error is -log(1-prediction)
      double calculateError(double z, Discriminator discriminator) {
        Vector generated = forward(z);
        double y = discriminator.forward(generated);
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
}