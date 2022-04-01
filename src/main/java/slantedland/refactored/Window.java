package slantedland.refactored;

import javax.swing.*;
import java.awt.Color;
import java.awt.Font;
import java.awt.Graphics;
import java.awt.image.BufferedImage;
import java.util.List;
import java.util.Random;

class Window extends JPanel {
    private final static Font FONT = new Font("Arial", Font.BOLD, 20);
    private final static int SAMPLES_PER_ROW = 3;
    private final static int SAMPLE_PADDING = 10;
    SlantedLandRefactored.Generator generatorInstance;
    List<Double> errors_discriminator;
    List<Double> errors_generator;
    Random random = new Random();
    
    public Window(SlantedLandRefactored.Generator generatorInstance, List<Double> errors_discriminator, List<Double> errors_generator) {
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
              double z = random.nextGaussian();
              double[] generated_image = generatorInstance.forward(z).data;
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
        ints[i] = (int) ((1d - input[i]) * 255d);
      }
      return ints;
    }
    
    @Override
    public void paint(Graphics graphics) {
      super.paint(graphics);
      
      double halfWidth = getWidth() / 2d - SAMPLE_PADDING * 2;
      double heightScale = 50;
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