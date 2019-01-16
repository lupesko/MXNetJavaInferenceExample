package MXNetDemo;

import MXNetDemo.ImageClassifier.ImageClassificationOutput;
import MXNetDemo.ImageClassifier.ImageClassifier;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

/**
 * Runs an image classification example
 *
 */
public class App 
{
    /**
     * Entry point for the image classification example, takes an input image path as an argument
     * @param args
     * @throws IOException
     */
    public static void main(String[] args) throws IOException
    {
        System.out.println("Initializing image classifier...");
        ImageClassifier classifier = new ImageClassifier(false);
        System.out.println("image classifier initialized.");

        String imagePath = "models/resnet-18/input.jpg";
        if (args.length == 0) {
            System.out.println("No image file argument supplied, using default models/resnet-18/input.jpg");
        } else {
            imagePath = args[0];
        }

        System.out.println("Loading image from file: " + imagePath + " ...");
        BufferedImage image = loadImageFromFile(imagePath);
        System.out.println("Image loaded successfully.");

        System.out.println("Invoking inference...");
        ImageClassificationOutput output = classifier.predict(image);
        System.out.format("Inference result: top class: %s, probability: %f%n", output.getClassName(), output.getClassProbability());

        System.out.println("Terminating...");
    }

    /**
     * Loads the image from file
     * @param inputImagePath
     * @return
     * @throws IOException
     */
    private static BufferedImage loadImageFromFile(String inputImagePath) throws IOException {
        BufferedImage image = null;
        try {
            image = ImageIO.read(new File(inputImagePath));
        } catch (IOException e) {
            System.err.println("Error when attempting to load an image. Error: " + e.getMessage());
            throw e;
        }

        return image;
    }
}
