package slantedland.refactored;

import java.util.Random;

public class Vector {
  private static Random random = new Random();
  
  public double[] data;
  
  public Vector(int size) {
    data = new double[size];
  }
  
  public Vector(double... data) {
    this.data = data;
  }
  
  public static Vector randomized(int count) {
    double[] newData = new double[count];
    for (int i = 0; i < count; i++) {
      newData[i] = random.nextGaussian();
    }
    return new Vector(newData);
  }
  
  public Vector inverse() {
    double[] newData = new double[data.length];
    for (int i = 0; i < data.length; i++) {
      newData[i] = -data[i];
    }
    return new Vector(newData);
  }
  
  public Vector multiply(double factor) {
    double[] newData = new double[data.length];
    for (int i = 0; i < data.length; i++) {
      newData[i] = data[i] * factor;
    }
    return new Vector(newData);
  }
  
  public double dot(Vector other) {
    if (data.length != other.data.length)
      throw new RuntimeException("Length of arrays doesn't match " + data.length + " != " + other.data.length);
    
    double sum = 0;
    for (int i = 0; i < data.length; i++) {
      sum += data[i] * other.data[i];
    }
    return sum;
  }
  
  public Vector dot(double a) {
    return this.multiply(a);
  }
  
  public Vector multiply(Vector other) {
    if (data.length != other.data.length)
      throw new RuntimeException("Length of arrays doesn't match!" + data.length + " != " + other.data.length);
    
    double[] newData = new double[data.length];
    for (int i = 0; i < data.length; i++) {
      newData[i] = data[i] * other.data[i];
    }
    
    return new Vector(newData);
  }
  
  public Vector sigmoid() {
    double[] newData = new double[data.length];
    for (int i = 0; i < data.length; i++) {
      newData[i] = Math.exp(data[i]) / (1.0 + Math.exp(data[i]));
    }
    return new Vector(newData);
  }
  
  public Vector subtractReverse(double other) {
    double[] result = new double[data.length];
    for (int i = 0; i < data.length; i++) {
      result[i] = other - data[i];
    }
    return new Vector(result);
  }
  
  public double sum() {
    double sum = 0;
    for (int i = 0; i < data.length; i++) {
      sum += data[i];
    }
    return sum;
  }
  
  public Vector add(double other) {
    double[] result = new double[data.length];
    for (int i = 0; i < data.length; i++) {
      result[i] = data[i] + other;
    }
    return new Vector(result);
  }
  
  public Vector add(Vector other) {
    double[] result = new double[data.length];
    for (int i = 0; i < data.length; i++) {
      result[i] = data[i] + other.data[i];
    }
    return new Vector(result);
  }
  
  public Vector subtract(Vector other) {
    if (data.length != other.data.length)
      throw new RuntimeException("Length of arrays doesn't match! " + data.length + " != " + other.data.length);
    
    double[] newData = new double[data.length];
    for (int i = 0; i < data.length; i++) {
      newData[i] = data[i] - other.data[i];
    }
    
    return new Vector(newData);
  }
  
  public Vector log() {
    double[] result = new double[data.length];
    for (int i = 0; i < data.length; i++) {
      result[i] = Math.log(data[i]);
    }
    return new Vector(result);
  }
}