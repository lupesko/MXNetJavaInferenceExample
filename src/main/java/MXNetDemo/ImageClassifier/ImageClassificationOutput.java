package MXNetDemo.ImageClassifier;

/**
 * Simple POJO class that represents image classification result
 */
public class ImageClassificationOutput {
    private float classProbability;
    private String className;

    /**
     * Initializes a new instance of ImageClassificationOutput
     * @param classProbability
     * @param className
     */
    public ImageClassificationOutput(final float classProbability, final String className) {
        this.classProbability = classProbability;
        this.className = className;
    }

    /**
     * Returns the class probability of this result
     * @return probability in the range [0 1]
     */
    public float getClassProbability() {
        return this.classProbability;
    }

    /**
     * Returns the class name of this result
     * @return class name
     */
    public String getClassName() {
        return this.className;
    }
}
