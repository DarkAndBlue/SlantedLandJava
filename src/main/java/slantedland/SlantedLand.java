package slantedland;

import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class SlantedLand {
  static Random rand = new Random();
  
  public static void main(String[] args) {
    startGan();
  }
  
//#  changes x input type between ndarray and float64
//  def sigmoid(x):
//    return np.exp(x) / (1.0 + np.exp(x))
  public static double[] sigmoid(double[] x) {
    double[] y = new double[x.length];
    for (int i = 0; i < x.length; i++) {
      y[i] = Math.exp(x[i]) / (1.0 + Math.exp(x[i]));
    }
    return y;
  }
  
  public static double sigmoid(double x) {
    return Math.exp(x) / (1.0 + Math.exp(x));
  }
  
//    # real samples
//  faces: list[ndarray] = [np.array([1, 0, 0, 1]),
//    np.array([0.9, 0.1, 0.2, 0.8]),
//    np.array([0.9, 0.2, 0.1, 0.8]),
//    np.array([0.8, 0.1, 0.2, 0.9]),
//    np.array([0.8, 0.2, 0.1, 0.9])]
  static double[][] faces = { { 1, 0, 0, 1 }, { 0.9, 0.1, 0.2, 0.8 }, { 0.9, 0.2, 0.1, 0.8 }, { 0.8, 0.1, 0.2, 0.9 }, { 0.8, 0.2, 0.1, 0.9 } };
  
//  noise: list[np.ndarray] = [np.random.randn(2, 2) for i in range(20)]
  /*
    [array([[ 1.30345511,  0.87838275], [-1.55296293,  1.37779997]]),
    array([[-2.26870357, -2.20717963], [-0.49189746, -0.65753657]]),
    array([[-0.80440499, -0.93580025], [-0.1639378 , -0.99063832]]),
    array([[-0.08447247, -0.9797766 ], [-1.83319705,  0.46227672]]),
    array([[ 2.56004247,  0.8807973 ], [-1.02426026, -0.17796001]]),
    array([[-0.18046342,  0.36269715], [-1.1387195 , -1.4920937 ]]),
    array([[-0.4265645 , -1.25812555], [ 1.39998131,  0.3174858 ]]),
    array([[-1.25982773, -1.07234258], [ 1.22973594,  0.82860369]]),
    array([[-1.82867334, -0.87734434], [ 0.14631145,  0.87300768]]),
    array([[1.23561225, 0.34139066], [0.3314154 , 0.33955404]]),
    array([[1.30521486, 1.16446244], [2.08742436, 1.42871303]]),
    array([[ 0.46582964,  2.46230089], [-0.86825156,  0.54095977]]),
    array([[ 0.95794467,  0.57709746], [ 0.20246825, -2.30295568]]),
    array([[ 0.32217008,  0.79256277], [-1.20979805, -0.85345128]]),
    array([[-0.63536707,  0.9800105 ], [ 0.07915123,  1.12833763]]),
    array([[-1.94996036,  0.13961783], [-1.01360346,  0.10096855]]),
    array([[-1.28524636,  0.4321207 ], [ 0.68793205,  1.13987595]]),
    array([[ 0.0744313 , -1.19580862], [ 1.08278883,  0.81317792]]),
    array([[ 0.41719021,  1.18373733], [-0.37367713,  1.26556795]]),
    array([[ 1.51587293,  0.1794817 ], [-0.99391503, -0.04169526]])]
   */
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
  
//  def generate_random_image():
//    return [np.random.random(), np.random.random(), np.random.random(), np.random.random()]
  public static double[] generate_random_image() {
    double[] x = new double[4];
    for (int i = 0; i < 4; i++) {
      x[i] = rand.nextGaussian();
    }
    return x;
  }
  
  static class np {
    public static double[] dot(double a, double[] b) {
      return multiply(a, b);
    }
    
    public static double dot(double a[], double[] b) {
      if (a.length != b.length)
        throw new RuntimeException("Length of arrays doesn't match " + a.length + " != " + b.length);
      
      double sum = 0;
      for (int i = 0; i < a.length; i++) {
        sum += a[i] * b[i];
      }
      return sum;
    }
    
    public static double[] log(double[] input) {
      double[] result = new double[input.length];
      for (int i = 0; i < input.length; i++) {
        result[i] = Math.log(result[i]);
      }
      return result;
    }
    
    public static double[] multiply(double a, double[] b) {
      double[] c = new double[b.length];
      for (int i = 0; i < b.length; i++) {
        c[i] = b[i] * a;
      }
      return c;
    }
    
    public static double[] multiply(double[] a, double[] b) {
      if (a.length != b.length)
        throw new RuntimeException("Length of arrays doesn't match!" + a.length + " != " + b.length);
      
      double[] c = new double[b.length];
      for (int i = 0; i < a.length; i++) {
        c[i] = a[i] * b[i];
      }
      
      return c;
    }
  }
  
  public static double[] subtractReverse(double a, double[] b) {
    double[] result = new double[b.length];
    for (int i = 0; i < b.length; i++) {
      result[i] = a - b[i];
    }
    return result;
  }
  
  public static double[] inverse(double[] input) {
    double[] result = new double[input.length];
    for (int i = 0; i < input.length; i++) {
      result[i] = -result[i];
    }
    return result;
  }
  
  public static double sum(double[] n) {
    double sum = 0;
    for (int i = 0; i < n.length; i++) {
      sum += n[i];
    }
    return sum;
  }
  
//  class Discriminator:
  static class Discriminator {
    //  weights: ndarray # [ 0.49671415 -0.1382643   0.64768854  1.52302986]
    double[] weights;
    //  biases: float64
    double bias;
    
//  def __init__(self):
//    # self.weights = np.array([0.0 for i in range(4)])
//    # self.bias = 0.0
//  self.weights = np.array([np.random.normal() for i in range(4)])
//  self.bias = np.random.normal()
    public Discriminator() {
      weights = new double[4];
      for (int i = 0; i < weights.length; i++) {
        weights[i] = rand.nextGaussian();
      }
      
      bias = rand.nextGaussian();
    }
    
    
//    # if x is float return value is ndarray, if x is ndarray return value is float64
//    def forward(self, x):
//      # Forward pass
//        return sigmoid(np.dot(x, self.weights) + self.bias)
    double[] forward(double x) {
      double[] a = np.dot(x, weights);
      for (int i = 0; i < a.length; i++) {
        a[i] += bias;
      }
      return sigmoid(a);
    }
    
    double forward(double[] x) {
      return sigmoid(np.dot(x, weights) + bias);
    }
    
//  def error_from_image(self, image: ndarray) -> float64:
//  prediction: float64 = self.forward(image)
//    # We want the prediction to be 1, so the error is -log(prediction)
//        return -np.log(prediction)
    double error_from_image(double[] image) {
      double prediction = forward(image);
      return -Math.log(prediction);
    }
    
//  def derivatives_from_image(self, image: ndarray) -> tuple:
//  prediction: float64 = self.forward(image)
//  derivatives_weights: ndarray = -image * (1 - prediction)
//  derivative_bias: float64 = -(1 - prediction)
//    return derivatives_weights, derivative_bias
    Object[] derivatives_from_image(double[] image) {
      double prediction = forward(image);
      double[] derivatives_weights = new double[4];
      for (int i = 0; i < derivatives_weights.length; i++) {
        derivatives_weights[i] = -image[i] * (1 - prediction);
      }
      double derivative_bias = -(1 - prediction);
      return new Object[] { derivatives_weights, derivative_bias };
    }
    
//  def update_from_image(self, x: ndarray):
//  ders: tuple = self.derivatives_from_image(x)
//  self.weights -= learning_rate * ders[0]
//  self.bias -= learning_rate * ders[1]
    void update_from_image(double[] x) {
      Object[] ders = derivatives_from_image(x);
      double[] derivatives_weights = (double[]) ders[0];
      double derivative_bias = (double) ders[1];
      for (int i = 0; i < weights.length; i++) {
        weights[i] -= learning_rate * derivatives_weights[i];
      }
      bias -= learning_rate * derivative_bias;
    }
    
//  def error_from_noise(self, noise: float) -> ndarray:
//  prediction: ndarray = self.forward(noise)
//    # We want the prediction to be 0, so the error is -log(1-prediction)
//        return -np.log(1 - prediction)  # ndarray
    double[] error_from_noise(double noise) {
      double[] prediction = forward(noise);
      return inverse(np.log(subtractReverse(1, prediction)));  // ndarray
    }
    
//  def derivatives_from_noise(self, noise: ndarray) -> tuple:
//  prediction: float64 = self.forward(noise)
//  derivatives_weights: ndarray = noise * prediction
//  derivative_bias: float64 = prediction
//        return derivatives_weights, derivative_bias
    Object[] derivatives_from_noise(double[] noise) {
      double prediction = forward(noise);
      double[] derivatives_weights = new double[4];
      for (int i = 0; i < derivatives_weights.length; i++) {
        derivatives_weights[i] = noise[i] * prediction;
      }
      double derivative_bias = prediction;
      return new Object[] { derivatives_weights, derivative_bias };
    }
    
//  def update_from_noise(self, noise: ndarray):
//  ders: tuple = self.derivatives_from_noise(noise)
//  self.weights -= learning_rate * ders[0]
//  self.bias -= learning_rate * ders[1]
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
  
//  class Generator:
  static class Generator {
//  weights: ndarray # [-0.23413696  1.57921282  0.76743473 -0.46947439]
    double[] weights;
    
//  biases: ndarray # [ 0.54256004 -0.46341769 -0.46572975  0.24196227]
    double[] biases;
    
//  def __init__(self):
//  self.weights = np.array([np.random.normal() for i in range(4)])
//  self.biases = np.array([np.random.normal() for i in range(4)])
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
    
//  def forward(self, z: float) -> ndarray:
//    # Forward pass
//        return sigmoid(z * self.weights + self.biases)
    double[] forward(double z) {
      double[] z_weights = new double[4];
      for (int i = 0; i < z_weights.length; i++) {
        z_weights[i] = z * weights[i] + biases[i];
      }
      return sigmoid(z_weights);
    }
    
//  def error(self, z: float, discriminator: Discriminator) -> float64:
//  x: ndarray = self.forward(z)
//    # We want the prediction to be 0, so the error is -log(1-prediction)
//  y: float64 = discriminator.forward(x)
//    return -np.log(y)
    double error(double z, Discriminator discriminator) {
      double[] x = forward(z);
      double y = discriminator.forward(x);
      return -Math.log(y);
    }
    
//  def derivatives(self, z: float, discriminator: Discriminator) -> tuple:
//  discriminator_weights: ndarray = discriminator.weights
//#  discriminator_bias: float64 = discriminator.bias
//  x: ndarray = self.forward(z)
//  y: float64 = discriminator.forward(x)
//  factor: ndarray = -(1 - y) * discriminator_weights * x * (1 - x)
//  derivatives_weights: ndarray = factor * z
//  derivative_bias: ndarray = factor
//        return derivatives_weights, derivative_bias
    Object[] derivatives(double z, Discriminator discriminator) {
      double[] discriminator_weights = discriminator.weights;
//      double discriminator_bias = discriminator.bias;
      double[] x = forward(z);
      double y = discriminator.forward(x);
      
      double a = -(1 - y);
      double[] b = np.multiply(a, discriminator_weights);
      b = np.multiply(b, x);
      double[] factor = np.multiply(b, subtractReverse(1, x));
      
      double[] derivatives_weights = new double[4];
      for (int i = 0; i < derivatives_weights.length; i++) {
        derivatives_weights[i] = factor[i] * z;
      }
      double[] derivative_bias = factor;
      return new Object[] { derivatives_weights, derivative_bias };
    }
    
//  def update(self, z: float, discriminator: Discriminator):
//  error_before: float64 = self.error(z, discriminator)
//  ders: tuple = self.derivatives(z, discriminator)
//  self.weights -= learning_rate * ders[0]
//  self.biases -= learning_rate * ders[1]
//  error_after: float64 = self.error(z, discriminator)
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
  
//    # Set random seed
//np.random.seed(42)
//  
//  # Hyperparameters
//  learning_rate: float = 0.01
  static double learning_rate = 0.01;
  
//  epochs: int = 1000
  static int epochs = 1000;
  
//    # The GAN
  public static void startGan() {
//  D = Discriminator()
//  G = Generator()
    Discriminator d = new Discriminator();
    Generator g = new Generator();


//# For the error plot
//  errors_discriminator: list[float64] = []
//  errors_generator: list[float64] = []
    List<Double> errors_discriminator = new ArrayList<>();
    List<Double> errors_generator = new ArrayList<>();

//    for epoch in range(epochs):
    for (int i = 0; i < epochs; i++) {

//  face: ndarray
//    for face in faces:  # faces = list of ndarrays
      for (double[] face : faces) {
//        # Update the discriminator weights from the real face
//        D.update_from_image(face)
        d.update_from_image(face);

//    # Pick a random number to generate a fake face
//  z: float = random.rand()
        double z = rand.nextGaussian();

//    # Calculate the discriminator error
//        errors_discriminator.append(sum(D.error_from_image(face) + D.error_from_noise(z)))
        double a = d.error_from_image(face);//float64
        double[] b = d.error_from_noise(z);//ndarray
        double[] c = new double[b.length];
        for (int j = 0; j < b.length; j++) {
          c[j] = a + b[j];
        }
        errors_discriminator.add(sum(c));

//    # Calculate the generator error
//        errors_generator.append(G.error(z, D))
        errors_generator.add(g.error(z, d));

//    # Build a fake face
//  noise: ndarray = G.forward(z)
        double[] noise = g.forward(z);

//    # Update the discriminator weights from the fake face
//        D.update_from_noise(noise)
        d.update_from_noise(noise);

//    # Update the generator weights from the fake face
//        G.update(z, D)
        g.update(z, d);
      }
    }
    
    new Window(g, errors_discriminator, errors_generator);
  }
  
  static class Window extends JPanel {
    private final static Font FONT = new Font("Arial", Font.BOLD, 20);
    private final static int SAMPLES_PER_ROW = 3;
    private final static int SAMPLE_PADDING = 10;
    Generator generatorInstance;
    List<Double> errors_discriminator;
    List<Double> errors_generator;
  
    public Window(Generator generatorInstance, List<Double> errors_discriminator, List<Double> errors_generator) {
      this.generatorInstance = generatorInstance;
      this.errors_discriminator = errors_discriminator;
      this.errors_generator = errors_generator;
      
      // Creating a JFrame to make the JPanel double buffered
      JFrame frame = new JFrame("Slanted Land Java");
      frame.setLayout(null);
      frame.setSize(1200, 600);
      frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
      frame.setVisible(true);
      frame.setLocationRelativeTo(null);
      frame.setBackground(new Color(117, 117, 117));
      frame.add(this);
      
      setSize(frame.getWidth(), frame.getHeight());
      
      setDoubleBuffered(true);
      
      JButton buttonSamples = new JButton("Press for samples");
      buttonSamples.addActionListener(e -> drawSamples());
      buttonSamples.setLocation(20, getHeight() - 80);
      buttonSamples.setSize(150, 30);
      frame.add(buttonSamples);
    }
    
    void drawSamples() {
      JFrame frame = new JFrame("samples");
      JPanel panel = new JPanel() {
        public void paint(Graphics graphics) {
          setDoubleBuffered(true);
          graphics.setColor(new Color(155, 155, 117));
          graphics.fillRect(0, 0, getWidth(), getHeight());
          int padding = 10;
          int size = getWidth() / SAMPLES_PER_ROW;
          for (int x = 0; x < SAMPLES_PER_ROW; x++) {
            for (int y = 0; y < SAMPLES_PER_ROW; y++) {
              double z = rand.nextGaussian();
              double[] generated_image = generatorInstance.forward(z);
              int[] pixels = doubleValuesToPixels(generated_image);
              
              BufferedImage image = new BufferedImage(2, 2, BufferedImage.TYPE_INT_RGB);
              for (int i = 0; i < pixels.length; i++) {
                image.setRGB(i % 2, i / 2, new Color(pixels[i], pixels[i], pixels[i]).getRGB());
              }
              graphics.drawImage(image,
                x * size + x * padding + padding,
                y * size + y * padding + padding,
                size - padding * 4,
                size - padding * 4,
                null
              );
            }
          }
        }
      };
      frame.add(panel);
      frame.setSize(600, 600);
      frame.setVisible(true);
      frame.setLocationRelativeTo(null);
    }
    
    int[] doubleValuesToPixels(double[] input) {
      int[] ints = new int[input.length];
      for (int i = 0; i < input.length; i++) {
        ints[i] = (int) (input[i] * 255d);
      }
      return ints;
    }
    
    @Override
    public void paint(Graphics graphics) {
      super.paint(graphics);
      
      double halfWidth = getWidth() / 2d - SAMPLE_PADDING * 2;
      double heightScale = getHeight() / 4;
      double size = halfWidth / errors_discriminator.size();
      graphics.setColor(Color.blue);
      
      int[] x = new int[errors_generator.size()];
      int[] y = new int[errors_generator.size()];
      for (int i = 0; i < errors_generator.size(); i++) {
        x[i] = (int) (i * size) + SAMPLE_PADDING;
        y[i] = invertY(errors_generator.get(i) * heightScale);
      }
      graphics.drawPolyline(x, y, x.length);
      
      x = new int[errors_discriminator.size()];
      y = new int[errors_discriminator.size()];
      for (int i = 0; i < errors_discriminator.size(); i++) {
        x[i] = (int) (i * size + halfWidth) + SAMPLE_PADDING * 3;
        y[i] = invertY(errors_discriminator.get(i) * heightScale);
      }
      graphics.drawPolyline(x, y, x.length);
      
      graphics.setColor(Color.black);
      graphics.setFont(FONT);
      graphics.drawString("Generator", SAMPLE_PADDING, 50);
      graphics.drawString("Discriminator", getWidth() / 2 + SAMPLE_PADDING * 2, 50);
    }
    
    int invertY(double y) {
      return (int) (getHeight() - y);
    }
  }
}