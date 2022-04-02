package mnistgan;

import slantedland.refactored.Vector;

import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.util.List;
import java.util.Random;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.ThreadLocalRandom;

public class MNISTGAN extends JFrame {
  /*
  INFO: Goal is it to write a GAN to generate images based on the MNIST dataset.
   */
  
  public static void main(String[] args) throws InterruptedException {
    new MNISTGAN();
  }
  
  long sleepTime = 10;
  
  public MNISTGAN() throws InterruptedException {
//    setTitle("MNIST GENERATION");
    setDefaultCloseOperation(EXIT_ON_CLOSE);
    setLayout(null);
    setSize(1800, 800);
    
    Scene scene = new Scene();
    scene.setSize(1800, 600);
    add(scene);
    
    setVisible(true);
    setLocationRelativeTo(null);
    
    JSlider slider = new JSlider();
    add(slider);
    slider.setSize(getWidth() - 20, 60);
    slider.setMaximum(120);
    slider.setMinimum(0);
    slider.setValue((int) sleepTime);
    slider.setLocation(0, scene.getY() + scene.getHeight());
    slider.addChangeListener(e -> sleepTime = slider.getValue());
  
    JSlider sliderInput = new JSlider();
    add(sliderInput);
    sliderInput.setSize(getWidth() - 20, 60);
    sliderInput.setMaximum(1000);
    sliderInput.setMinimum(0);
    sliderInput.setValue(sliderInput.getMaximum() / 2);
    sliderInput.setLocation(0, slider.getY() + slider.getHeight());
    sliderInput.addChangeListener(e -> scene.input = (double) sliderInput.getValue() / sliderInput.getMaximum());
  
    JSlider sliderLearningRate = new JSlider();
    add(sliderLearningRate);
    sliderLearningRate.setSize(getWidth() - 20, 60);
    sliderLearningRate.setMaximum(10000);
    sliderLearningRate.setMinimum(0);
    sliderLearningRate.setValue(sliderLearningRate.getMaximum() / 4);
    sliderLearningRate.setLocation(0, sliderInput.getY() + sliderInput.getHeight());
    sliderLearningRate.addChangeListener(e -> Scene.learning_rate = (double) sliderLearningRate.getValue() / (sliderLearningRate.getMaximum() * 8d));
    
    new Thread(() -> {
      while (true) {
        try {
          Thread.sleep(32);
        } catch (InterruptedException e) {
          e.printStackTrace();
        }
        setTitle("learningRate " + Scene.learning_rate);
        scene.repaint();
      }
    }).start();
    
    while (true) {
      if(sleepTime != 0)
        Thread.sleep(sleepTime);
      scene.updateNeuralNetworks();
    }
  }
  
  class Scene extends JPanel {
    static Random random = new Random();
    static int imageSize = 16;
    static double learning_rate = 0.03;
    static int targetsPerEpoch = 5;
    Discriminator discriminator;
    Generator generator;
    List<Double> discriminatorErrors = new CopyOnWriteArrayList<>();
    List<Double> generatorErrors = new CopyOnWriteArrayList<>();
    double input;
    
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
      
      Vector forward = generator.predict(input);
      
      for (int x = 0; x < imageSize; x++) {
        for (int y = 0; y < imageSize; y++) {
          int i = x + y * imageSize;
          int brightness = (int) (forward.data[i] * 255d);
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
      
      Double[] values = list.toArray(new Double[0]);
      
      int[] x = new int[values.length];
      int[] y = new int[values.length];
      
      for (int i = 0; i < values.length; i++) {
        double value = values[i];
        
        x[i] = (int) (i * sizeX) + xOffset;
        y[i] = (int) ((value - yOffset) / yScale) + getHeight() / 2;
      }
      
      graphics.drawPolyline(x, y, values.length);
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
        discriminator.trainOnTarget(target);
        
        // Pick a random number to generate a fake face
        double noise = random.nextGaussian();
        
        // Calculate the discriminator error
        double targetPrediction = discriminator.calculateError(target, noise, generator);
        discriminatorErrors.add(targetPrediction);
        
        // Calculate the generator error
        generatorErrors.add(generator.calculateError(noise, discriminator));
        
        // Build a fake face
        Vector generated = generator.predict(noise);
        
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
      graphics.setColor(Color.white);
      graphics.fillRect(0, 0, imageSize, imageSize);
      graphics.setColor(Color.black);
      int x = 0;
      int width = imageSize;
      if (Math.random() > 0.5) {
        x = imageSize;
        width = 0;
      }
      graphics.drawLine(x, 0, width, imageSize);
      
      double[] values = new double[imageSize * imageSize];
      for (int j = 0; j < imageSize * imageSize; j++) {
        int color = image.getRGB(j % imageSize, j / imageSize);
        values[j] = new Color(color).getRed() / 255d;
      }
      return new Vector(values);
    }
    
    static class Discriminator {
      Vector weights;
      double bias;
      
      public Discriminator() {
        weights = Vector.randomized(imageSize * imageSize);
        bias = random.nextGaussian();
      }
      
      // Forward pass
      double predict(Vector input) {
        return sigmoid(input.dot(weights) + bias);
      }
      
      void trainOnTarget(Vector targets) {
        // should be high 1
        double prediction = predict(targets);
        
        // dw = -target * (1 - prediction)
        Vector derivatives_weights = targets.inverse().multiply(1 - prediction);
        double derivative_bias = -(1 - prediction);
        
        weights = weights.subtract(derivatives_weights.multiply(learning_rate));
        bias -= learning_rate * derivative_bias;
      }
      
      void trainOnFake(Vector generated) {
        // prediction on how real the generated image is
        // should be low 0
        double prediction = predict(generated);
        
        Vector derivatives_weights = generated.multiply(prediction);
        double derivative_bias = prediction;
        
        weights = weights.subtract(derivatives_weights.multiply(learning_rate));
        bias -= learning_rate * derivative_bias;
      }
      
      public double calculateError(Vector target, double noise, Generator generator) {
        Vector generated = generator.predict(noise);
        double predictionFake = predict(generated);
        double predictionReal = 1 - predict(target);
        
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
      Vector predict(double noise) {
        Vector z_weights = weights.multiply(noise).add(biases);
        return z_weights.sigmoid();
      }
      
      // We want the prediction to be 0, so the error is -log(1-prediction)
      double calculateError(double z, Discriminator discriminator) {
        Vector generated = predict(z);
        double y = discriminator.predict(generated);
        return -Math.log(y);
      }
      
      void train(double noiseInput, Discriminator discriminator) {
        Vector discriminatorWeights = discriminator.weights;
        Vector generatorPrediction = predict(noiseInput);
        double discriminatorPrediction = discriminator.predict(generatorPrediction);
        
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