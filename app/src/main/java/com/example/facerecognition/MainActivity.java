package com.example.facerecognition;

import android.content.Context;
import android.content.Intent;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Bundle;
import android.widget.ImageButton;
import android.widget.ImageView;
import android.widget.TextView;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

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
    public Bitmap originalBitmap, textBitmap;

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

        initComponents();
    }

    private void initComponents() {
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
        FileInputStream fis = new FileInputStream(descriptor.getFileDescriptor());
        FileChannel channel = fis.getChannel();
        long startOffset = descriptor.getStartOffset();
        long declaredLength = descriptor.getDeclaredLength();
        return channel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    private TensorImage loadFile(final Bitmap bitmap, TensorImage inputImageBuffer) {
        inputImageBuffer.load(bitmap);
        int cropSize = Math.min(bitmap.getWidth(), bitmap.getHeight());

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
    }
}