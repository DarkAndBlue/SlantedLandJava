package slantedland.refactored;

class Numpy {
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
}