package com.example.facerecognition;

import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Bundle;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import androidx.appcompat.app.AppCompatActivity;

public class MainActivity extends AppCompatActivity {
    private static final float IMAGE_MEAN = 0.0f;
    private static final float IMAGE_STD = 1.0f;
    public static Bitmap cropped;
    public Bitmap originalBitmap, textBitmap;
    protected Interpreter tflite;
    Uri imageUri;
    TextView result;
    ImageView originalImage, testImage;
    Button verify;
    float[][] originalEmbedding = new float[1][128];
    float[][] testEmbedding = new float[1][128];
    private int imageSizeX;
    private int imageSizeY;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
    }
}