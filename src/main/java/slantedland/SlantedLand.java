package slantedland;

import java.util.Random;

public class SlantedLand {
  static Random rand = new Random();
  
  //# TODO: changes x input type between ndarray and float64
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
  
  //  class Discriminator:
  class Discriminator {
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
    
    
    //    # TODO: x is a ndarray and some times a float. Check which one get called from where
//    # TODO: if x is float return value is ndarray, if x is ndarray return value is float64
//    def forward(self, x):
//      # Forward pass
//        return sigmoid(np.dot(x, self.weights) + self.bias)  # TODO: changes classes depending on x (float64 and ndarray)
    double[] forward(double x) {
      throw new RuntimeException("not implemented");
//        return sigmoid(np.dot(x, self.weights) + self.bias)
    }
    
    double forward(double[] x) {
      throw new RuntimeException("not implemented");
//        return sigmoid(np.dot(x, self.weights) + self.bias)
    }
    
    //  def error_from_image(self, image: ndarray) -> float64:
//  prediction: float64 = self.forward(image)
//    # We want the prediction to be 1, so the error is -log(prediction)
//        return -np.log(prediction)
    double error_from_image(double[] image) {
      double prediction = forward(image);
      throw new RuntimeException("not implemented");
      //        return -np.log(prediction)
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
      throw new RuntimeException("not implemented");
      //        return -np.log(1 - prediction)  # ndarray
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
    
    //  
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
  class Generator {
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
      throw new RuntimeException("not implemented");
//      double[] factor = -(1 - y) * discriminator_weights * x * (1 - x);
//      double[] derivatives_weights = new double[4];
//      for (int i = 0; i < derivatives_weights.length; i++) {
//        derivatives_weights[i] = factor[i] * z;
//      }
//      double[] derivative_bias = factor;
//      return new Object[] { derivatives_weights, derivative_bias };
    }
    
    //  def update(self, z: float, discriminator: Discriminator):
//  error_before: float64 = self.error(z, discriminator)
//  ders: tuple = self.derivatives(z, discriminator)
//  self.weights -= learning_rate * ders[0]
//  self.biases -= learning_rate * ders[1]
//  error_after: float64 = self.error(z, discriminator)
    void update(double z, Discriminator discriminator) {
      double error_before = error(z, discriminator);
      Object[] ders = derivatives(z, discriminator);
      double[] derivatives_weights = (double[]) ders[0];
      double[] derivative_bias = (double[]) ders[1];
      for (int i = 0; i < weights.length; i++) {
        weights[i] -= learning_rate * derivatives_weights[i];
      }
      for (int i = 0; i < weights.length; i++) {
        biases[i] -= learning_rate * derivative_bias[i];
      }
      double error_after = error(z, discriminator);
    }
  }
  
  //    
//    # Set random seed
//np.random.seed(42)
//  
//  # Hyperparameters
//  learning_rate: float = 0.01
  double learning_rate;
//  epochs: int = 1000
//    
//    # The GAN
//  D = Discriminator()
//  G = Generator()
//
//# For the error plot
//  errors_discriminator: list[float64] = []
//  errors_generator: list[float64] = []
//    
//    for epoch in range(epochs):
//  face: ndarray
//    for face in faces:  # faces = list of ndarrays
//        # Update the discriminator weights from the real face
//        D.update_from_image(face)
//    
//    # Pick a random number to generate a fake face
//  z: float = random.rand()
//    
//    # Calculate the discriminator error
//        errors_discriminator.append(sum(D.error_from_image(face) + D.error_from_noise(z)))
//    
//    # Calculate the generator error
//        errors_generator.append(G.error(z, D))
//    
//    # Build a fake face
//  noise: ndarray = G.forward(z)
//    
//    # Update the discriminator weights from the fake face
//        D.update_from_noise(noise)
//    
//    # Update the generator weights from the fake face
//        G.update(z, D)
}