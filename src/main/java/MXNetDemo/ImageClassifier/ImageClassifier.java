package MXNetDemo.ImageClassifier;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.apache.mxnet.javaapi.*;
import org.apache.mxnet.infer.javaapi.Predictor;
import org.apache.mxnet.javaapi.Shape;

/**
 * This class encapsulates image classifier based on a deep learning model built with MXNet
 */
public class ImageClassifier {

    // Data members
    private Predictor imageClassifier;
    private List<String> classNames;
    private final int inputImageWidth, inputImageHeight;
    private static final String modelPathPrefix = "models/resnet-18/resnet-18";
    private static final String classNamesFilePath = "models/resnet-18/synset.txt";

    /**
     * Initializes a new instance of the ImageClassifier class
     * @param useGpu
     * @throws FileNotFoundException
     * @throws IOException
     */
    public ImageClassifier(boolean useGpu) throws FileNotFoundException, IOException {

        // Set the appropriate context
        Context context = Context.cpu();
        if (useGpu) {
            context = Context.gpu();
        }
        List<Context> inferenceContext = Arrays.asList(context);

        // Initialize the input descriptor
        inputImageWidth = inputImageHeight = 224;
        List<DataDesc> inputDesc = Arrays.asList(
                new DataDesc("data", new Shape(new int[]{1, 3, inputImageHeight, inputImageWidth}), DType.Float32(), "NCHW"));

        // Initialize the image classifier
        this.imageClassifier = new Predictor(modelPathPrefix, inputDesc, inferenceContext, 0);

        // Load class names into memory
        this.classNames = loadClassNames(classNamesFilePath);
    }

    /**
     * Predicts the classification output
     * @param image
     * @return
     */
    public ImageClassificationOutput predict(BufferedImage image) {
        // Pre-process input image
        image = resizeImage(image);
        float[] inputBuffer = imageToFloatBuffer(image);

        // predict
        float[] predictedClassProbabilities = imageClassifier.predict(new float[][]{ inputBuffer })[0];
        int maxClassIndex = getMaxIndex(predictedClassProbabilities);
        float maxProbability = predictedClassProbabilities[maxClassIndex];
        String maxClassName = classNames.get(maxClassIndex);

        return new ImageClassificationOutput(maxProbability, maxClassName);
    }

    /**
     * Loads the model's class names from the supplied file
     * @param classNamesFilePath
     * @return
     * @throws FileNotFoundException
     * @throws IOException
     */
    private static List<String> loadClassNames(String classNamesFilePath) throws FileNotFoundException, IOException {
        List<String> classNames = new ArrayList<>();

        try (BufferedReader reader = new BufferedReader(new FileReader(classNamesFilePath))) {
            String line = reader.readLine();
            while (line != null) {
                classNames.add(line);
                line = reader.readLine();
            }
        }

        return classNames;
    }

    /**
     * Resize the current image to the size the model expects
     * @param inputImage Buffered image
     * @return a resized bufferedImage
     */
    private BufferedImage resizeImage(BufferedImage inputImage) {
        BufferedImage resizedImage = new BufferedImage(this.inputImageWidth, this.inputImageHeight, BufferedImage.TYPE_INT_RGB);
        Graphics2D graphics = resizedImage.createGraphics();
        graphics.drawImage(inputImage, 0, 0, this.inputImageWidth, this.inputImageHeight, null);
        graphics.dispose();

        return resizedImage;
    }

    /**
     * Converts the supplied image into a single dimensional float buffer
     * @param inputImage
     * @return the single dimensional float array
     */
    private float[] imageToFloatBuffer(BufferedImage inputImage) {
        // Get height and width of the image
        int width = inputImage.getWidth();
        int height = inputImage.getHeight();

        // get an array of integer pixels in the default RGB color mode
        int[] pixels = inputImage.getRGB(0, 0, width, height, null, 0, width);

        // 3 times height and width for R,G,B channels
        float[] buffer = new float[3 * height * width];

        // copy pixels to array vertically
        int row = 0;
        while (row < height) {
            int col = 0;
            // copy pixels to array horizontally
            while (col < width) {
                int rgb = pixels[row * width + col];
                // getting red color
                buffer[row * width + col] = (rgb >> 16) & 0xFF;
                // getting green color
                buffer[height * width + row * width + col] = (rgb >> 8) & 0xFF;
                // getting blue color
                buffer[2 * height * width + row * width + col] = rgb & 0xFF;
                col += 1;
            }
            row += 1;
        }
        inputImage.flush();

        return buffer;
    }

    /**
     * Returns the index of the array element with the biggest value
     * @param array
     * @return
     */
    private int getMaxIndex(float[] array) {
        int maxIndex = 0;
        for (int i = 1; i < array.length; i++) {
            if (array[i] > array[maxIndex]) {
                maxIndex = i;
            }
        }

        return maxIndex;
    }
}
