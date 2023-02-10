package com.example.facerecognition;

import android.content.Context;
import android.content.Intent;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.graphics.Rect;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.widget.ImageButton;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import com.google.mlkit.vision.common.InputImage;
import com.google.mlkit.vision.face.Face;
import com.google.mlkit.vision.face.FaceDetection;
import com.google.mlkit.vision.face.FaceDetector;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.common.TensorOperator;
import org.tensorflow.lite.support.common.ops.NormalizeOp;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.support.image.ops.ResizeWithCropOrPadOp;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;

public class MainActivity extends AppCompatActivity {
    private static final float IMAGE_MEAN = 0.0f;
    private static final float IMAGE_STD = 1.0f;

    public static Bitmap cropped;
    public Bitmap originalBitmap, testBitmap;

    protected Interpreter tflite;

    Uri imageUri;

    TextView result;
    ImageView originalImage, testImage;
    ImageButton verify;

    float[][] originalEmbedding = new float[1][128];
    float[][] testEmbedding = new float[1][128];

    private int imageSizeX;
    private int imageSizeY;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        result = findViewById(R.id.result);
        originalImage = findViewById(R.id.originalImage);
        testImage = findViewById(R.id.testImage);
        verify = findViewById(R.id.verify);

        try {
            tflite = new Interpreter(loadModelFile(this));
        } catch (Exception e) {
            e.printStackTrace();
        }

        originalImage.setOnClickListener(v -> {
            Intent intent = new Intent().setType("image/*").setAction(Intent.ACTION_GET_CONTENT);
            startActivityForResult(Intent.createChooser(intent, "Select image"), 12);
        });

        testImage.setOnClickListener(v -> {
            Intent intent = new Intent().setType("image/*").setAction(Intent.ACTION_GET_CONTENT);
            startActivityForResult(Intent.createChooser(intent, "Select image"), 13);
        });

        verify.setOnClickListener(v -> {
            double distance = calculateDistance(originalEmbedding, testEmbedding);

            if (distance < 6.0) result.setText(R.string.samefaces);
            else result.setText(R.string.diffaces);
        });
    }

    private double calculateDistance(float[][] originalEmbedding, float[][] testEmbedding) {
        double sum = 0.0;

        for (int i = 0; i < 128; i++) {
            sum = sum + Math.pow(originalEmbedding[0][1] - testEmbedding[0][1], 2.0);
        }
        return Math.sqrt(sum);
    }

    private MappedByteBuffer loadModelFile(Context context) throws IOException {
        AssetFileDescriptor descriptor = context.getAssets().openFd("Qfacenet.tflite");
        FileInputStream fis;
        FileChannel channel = null;
        long startOffset = descriptor.getStartOffset();
        long declaredLength = descriptor.getDeclaredLength();
        try {
            fis = new FileInputStream(descriptor.getFileDescriptor());
            channel = fis.getChannel();
        } catch (Exception e) {
            e.printStackTrace();
        }
        assert channel != null;
        return channel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    private TensorImage loadFile(final Bitmap BTIMAP, TensorImage inputImageBuffer) {
        inputImageBuffer.load(BTIMAP);
        int cropSize = Math.min(BTIMAP.getWidth(), BTIMAP.getHeight());

        ImageProcessor processor = new ImageProcessor.Builder().add(new ResizeWithCropOrPadOp(cropSize, cropSize))
                .add(new ResizeOp(imageSizeX, imageSizeY, ResizeOp.ResizeMethod.NEAREST_NEIGHBOR))
                .add(getPreprocessNormalizeOp()).build();
        return processor.process(inputImageBuffer);
    }

    private TensorOperator getPreprocessNormalizeOp() {
        return new NormalizeOp(IMAGE_MEAN, IMAGE_STD);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);

        if (requestCode == 12 && resultCode == RESULT_OK && data != null) {
            imageUri = data.getData();

            try {
                originalBitmap = MediaStore.Images.Media.getBitmap(getContentResolver(), imageUri);
                originalImage.setImageBitmap(originalBitmap);
                detectFace(originalBitmap, "Original Image");
            } catch (IOException ioe) {
                ioe.printStackTrace();
            }
        }

        if (requestCode == 13 && resultCode == RESULT_OK && data != null) {
            imageUri = data.getData();

            try {
                testBitmap = MediaStore.Images.Media.getBitmap(getContentResolver(), imageUri);
                testImage.setImageBitmap(testBitmap);
                detectFace(testBitmap, "Test Image");
            } catch (IOException ioe) {
                ioe.printStackTrace();
            }
        }
    }

    private void detectFace(final Bitmap BITMAP, final String IMAGE_TYPE) {
        final InputImage INPUTIMAGE = InputImage.fromBitmap(BITMAP, 0);
        FaceDetector faceDetector = FaceDetection.getClient();
        faceDetector.process(INPUTIMAGE).addOnSuccessListener(faces -> {
            for (Face face : faces) {
                Rect bounds = face.getBoundingBox();
                cropped = Bitmap.createBitmap(BITMAP, bounds.left, bounds.top, bounds.width(), bounds.height());
                getEmbeddings(cropped, IMAGE_TYPE);
            }
        }).addOnFailureListener(e -> Toast.makeText(getApplicationContext(), e.getMessage(), Toast.LENGTH_LONG).show());
    }

    private void getEmbeddings(Bitmap bitmap, String imageType) {
        TensorImage inputImageBuffer;
        float[][] embedding = new float[1][128];
        int imageTensorIndex = 0;
        int[] imageShape = tflite.getInputTensor(imageTensorIndex).shape();
        imageSizeX = imageShape[1];
        imageSizeY = imageShape[2];
        DataType imageDataType = tflite.getInputTensor(imageTensorIndex).dataType();
        inputImageBuffer = new TensorImage(imageDataType);
        inputImageBuffer = loadFile(bitmap, inputImageBuffer);

        tflite.run(inputImageBuffer.getBuffer(), embedding);

        if (imageType.equals("Original Image")) {
            originalEmbedding = embedding;
        } else if (imageType.equals("Test Image")) {
            testEmbedding = embedding;
        }
    }
}