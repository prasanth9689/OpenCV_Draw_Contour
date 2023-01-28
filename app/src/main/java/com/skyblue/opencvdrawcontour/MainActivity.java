package com.skyblue.opencvdrawcontour;

import static com.skyblue.opencvdrawcontour.OpenCvColorConstants.blue;
import static org.opencv.android.CameraRenderer.LOGTAG;
import static org.opencv.imgproc.Imgproc.contourArea;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;

import android.Manifest;
import android.content.Context;
import android.content.pm.PackageManager;
import android.os.Bundle;
import android.util.Log;
import android.view.SurfaceView;
import android.view.View;
import android.view.Window;
import android.view.WindowManager;
import android.widget.Toast;

import com.skyblue.opencvdrawcontour.databinding.ActivityMainBinding;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.JavaCameraView;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.List;

public class MainActivity extends AppCompatActivity {
    private ActivityMainBinding binding;

    private static final int MY_CAMERA_REQUEST_CODE = 1;
    private static final int MY_STORAGE_REQUEST_CODE = 2;
    private Context context = this;

    int activeCamera = CameraBridgeViewBase.CAMERA_ID_BACK;

    private Mat mRGBA , mOutputRGBA;

    private List<MatOfPoint> contours = new ArrayList<MatOfPoint>();

    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status){
                case LoaderCallbackInterface.SUCCESS:{
                    Log.v(LOGTAG, "OpenCV Loaded");
                    binding.mOpenCvCameraView.enableView();
                }break;
                default:{
                    super.onManagerConnected(status);
                }break;
            }
            super.onManagerConnected(status);
        }
    };

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        FULL_SCREEN_REQUEST();
        binding = ActivityMainBinding.inflate(getLayoutInflater());
        View view = binding.getRoot();
        setContentView(view);

        if (ActivityCompat.checkSelfPermission(this, android.Manifest.permission.CAMERA)!= PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.CAMERA}, MY_CAMERA_REQUEST_CODE);
        }else {
            //Toast.makeText(context, "Camera permission already granted", Toast.LENGTH_SHORT).show();
            storagePermission();
        }

        initActivity();
    }

    private void initializeCamera() {
        binding.mOpenCvCameraView.setCameraPermissionGranted();
        binding.mOpenCvCameraView.setCameraIndex(activeCamera);
        binding.mOpenCvCameraView.setVisibility(SurfaceView.VISIBLE);
        binding.mOpenCvCameraView.setMaxFrameSize(1250,720); // binding.mOpenCvCameraView.setMaxFrameSize(3120,4160); (OR 1250 x 720)
        binding.mOpenCvCameraView.setCvCameraViewListener(cvCameraViewListener);
        binding.mOpenCvCameraView.setEnabled(true);
        setDisplayOrientation(binding.mOpenCvCameraView, 90);
    }

    protected void setDisplayOrientation(JavaCameraView camera, int angle){
        Method downPolymorphic;
        try
        {
            downPolymorphic = camera.getClass().getMethod("setDisplayOrientation", new Class[] { int.class });
            if (downPolymorphic != null)
                downPolymorphic.invoke(camera, new Object[] { angle });
        }
        catch (Exception e1)
        {
            e1.printStackTrace();
        }
    }

    private CameraBridgeViewBase.CvCameraViewListener2 cvCameraViewListener = new CameraBridgeViewBase.CvCameraViewListener2() {
        @Override
        public void onCameraViewStarted(int width, int height) {
            mRGBA = new Mat(height, width, CvType.CV_8UC4);
            mOutputRGBA = new Mat(height, width, CvType.CV_8UC4);
        }

        @Override
        public void onCameraViewStopped() {
            mRGBA.release();
            mOutputRGBA.release();
        }

        @Override
        public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
            mRGBA = inputFrame.rgba();
            if (Math.random()>0.80) {
                findSquares(inputFrame.rgba().clone(), contours);
            }
            mOutputRGBA = inputFrame.rgba().clone();

            for (int contourIdx = 0; contourIdx < contours.size(); contourIdx++)
            {
               // Imgproc.drawContours(mRGBA, contours, contourIdx, new Scalar(0,0,255), -1);
                Imgproc.drawContours(mRGBA, contours, 0, blue, 3);
            }

            //Imgproc.drawContours(mRGBA, contours, 0, blue, 3);
            return mRGBA;
        }

        int N = 11;

        // helper function: finds a cosine of angle between vectors, from pt0->pt1 and from pt0->pt2
        double angle(Point pt1, Point pt2, Point pt0 ) {
            double dx1 = pt1.x - pt0.x;
            double dy1 = pt1.y - pt0.y;
            double dx2 = pt2.x - pt0.x;
            double dy2 = pt2.y - pt0.y;
            return (dx1*dx2 + dy1*dy2)/Math.sqrt((dx1*dx1 + dy1*dy1)*(dx2*dx2 + dy2*dy2) + 1e-10);
        }

        // returns sequence of squares detected on the image. the sequence is stored in the specified memory storage
        void findSquares(@NonNull Mat image, @NonNull List<MatOfPoint> squares )
        {
            new Thread(new Runnable() {
                @Override
                public void run() {
                    squares.clear();

                    Mat smallerImg=new Mat(new Size(image.width()/2, image.height()/2),image.type());

                    Mat gray=new Mat(image.size(),image.type());
                    Mat gray0=new Mat(image.size(),CvType.CV_8U);

                    Imgproc.pyrDown(image, smallerImg, smallerImg.size()); // down-scale and upscale the image to filter out the noise
                    Imgproc.pyrUp(smallerImg, image, image.size());

                    for( int c = 0; c < 3; c++ )  // find squares in every color plane of the image
                    {
                        extractChannel(image, gray, c);

                        for( int l = 1; l < N; l++ ) // try several threshold levels
                        {
                            Imgproc.threshold(gray, gray0, (l+1)*255/N, 255, Imgproc.THRESH_BINARY);
                            List<MatOfPoint> contours=new ArrayList<MatOfPoint>();
                            // find contours and store them all as a list
                            Imgproc.findContours(gray0, contours, new Mat(), Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE);
                            MatOfPoint approx=new MatOfPoint();

                            for( int i = 0; i < contours.size(); i++ ) // test each contour
                            {
                                // approximate contour with accuracy proportional to the contour perimeter
                                approx = approxPolyDP(contours.get(i),  Imgproc.arcLength(new MatOfPoint2f(contours.get(i).toArray()), true)*0.02, true);
                                // square contours should have 4 vertices after approximation, relatively large area (to filter out noisy contours) and be convex.
                                // Note: absolute value of an area is used because, area may be positive or negative - in accordance with the contour orientation
                                if( approx.toArray().length == 4 &&
                                        Math.abs(contourArea(approx)) > 1000 &&
                                        Imgproc.isContourConvex(approx) )
                                {
                                    double maxCosine = 0;

                                    for( int j = 2; j < 5; j++ )
                                    {
                                        // find the maximum cosine of the angle between joint edges
                                        double cosine = Math.abs(angle(approx.toArray()[j%4], approx.toArray()[j-2], approx.toArray()[j-1]));
                                        maxCosine = Math.max(maxCosine, cosine);
                                    }
                                    // if cosines of all angles are small, (all angles are ~90 degree) then write quandrange, vertices to resultant sequence
                                    if( maxCosine < 0.3 )
                                        squares.add(approx);
                                }
                            }
                        }
                    }
                }
            }).start();
        }

        void extractChannel(Mat source, Mat out, int channelNum) {
            List<Mat> sourceChannels=new ArrayList<Mat>();
            List<Mat> outChannel=new ArrayList<Mat>();
            Core.split(source, sourceChannels);
            outChannel.add(new Mat(sourceChannels.get(0).size(),sourceChannels.get(0).type()));
            Core.mixChannels(sourceChannels, outChannel, new MatOfInt(channelNum,0));
            Core.merge(outChannel, out);
        }
        MatOfPoint approxPolyDP(MatOfPoint curve, double epsilon, boolean closed) {
            MatOfPoint2f tempMat=new MatOfPoint2f();
            Imgproc.approxPolyDP(new MatOfPoint2f(curve.toArray()), tempMat, epsilon, closed);
            return new MatOfPoint(tempMat.toArray());
        }
    };

    @Override
    public void onPause(){ super.onPause(); if (binding.mOpenCvCameraView != null){ binding.mOpenCvCameraView.disableView(); }}

    @Override
    public void onDestroy(){ super.onDestroy(); if (binding.mOpenCvCameraView != null){ binding.mOpenCvCameraView.disableView(); }}

    @Override
    public void onResume(){
        super.onResume();
        if (!OpenCVLoader.initDebug()){
            Log.d(LOGTAG, "OpenCV not found, initializing");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION, this, mLoaderCallback);
        }else {
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    }


    private void initActivity() {
        initializeCamera();
    }

    private void storagePermission() {
        if (ActivityCompat.checkSelfPermission(this, Manifest.permission.READ_EXTERNAL_STORAGE)!= PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.READ_EXTERNAL_STORAGE}, MY_STORAGE_REQUEST_CODE);
        }else {
            //Toast.makeText(context, "Storage Permission already granted", Toast.LENGTH_SHORT).show();
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        switch (requestCode) {
            case MY_CAMERA_REQUEST_CODE: {
                if (grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                    storagePermission();
                    //Toast.makeText(context, "Camera Permission Granted", Toast.LENGTH_SHORT).show();
                } else {
                    Toast.makeText(context, "Camera Permission Denied", Toast.LENGTH_SHORT).show();
                }
            };

            case MY_STORAGE_REQUEST_CODE:{
                if (grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                    //Toast.makeText(context, "Storage Permission Granted", Toast.LENGTH_SHORT).show();
                } else {
                    Toast.makeText(context, "Storage Permission Denied", Toast.LENGTH_SHORT).show();
                }
            }
        }
    }

    private void FULL_SCREEN_REQUEST() {
        requestWindowFeature(Window.FEATURE_NO_TITLE);
        getWindow().setFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN,
                WindowManager.LayoutParams.FLAG_FULLSCREEN);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
    }
}