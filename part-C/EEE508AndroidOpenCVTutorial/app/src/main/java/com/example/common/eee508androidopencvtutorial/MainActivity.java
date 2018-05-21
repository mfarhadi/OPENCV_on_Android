package com.example.common.eee508androidopencvtutorial;

import android.graphics.Color;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.view.MotionEvent;
import android.view.SurfaceView;
import android.view.View;
import android.view.WindowManager;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import android.view.View.OnTouchListener;
import android.widget.TextView;

import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.calib3d.Calib3d;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.DMatch;
import org.opencv.core.KeyPoint;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.MatOfDMatch;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.features2d.DescriptorExtractor;
import org.opencv.features2d.DescriptorMatcher;
import org.opencv.features2d.FeatureDetector;
import org.opencv.features2d.Features2d;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.Vector;

import static org.opencv.calib3d.Calib3d.RANSAC;
import static org.opencv.calib3d.Calib3d.findHomography;
import static org.opencv.features2d.DescriptorMatcher.*;
import static org.opencv.imgproc.Imgproc.circle;
import static org.opencv.imgproc.Imgproc.line;

public class MainActivity extends AppCompatActivity implements OnTouchListener,CvCameraViewListener2{
    private CameraBridgeViewBase mOpenCvCameraView;
    private Mat mRgba;
    private Mat mgray;
    private Mat image_base;
    double x=-1;
    double y=-1;
    int activate;
    private Scalar mBlobColorRgba;
    private Scalar mBlobColorHsv;
    private MatOfKeyPoint keypoints1;

    TextView touch_coordinates;
    TextView touch_color;

    Scalar RED = new Scalar(255, 0, 0);
    Scalar GREEN = new Scalar(0, 255, 0);
    FeatureDetector detector;
    DescriptorExtractor descriptor;
    DescriptorMatcher matcher;
    Mat descriptors1;



    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                {
                    //Log.i(TAG, "OpenCV loaded successfully");
                    mOpenCvCameraView.enableView();
                    mOpenCvCameraView.setOnTouchListener(MainActivity.this);
                } break;
                default:
                {
                    super.onManagerConnected(status);
                } break;
            }
        }
    };


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        touch_coordinates = (TextView) findViewById(R.id.touch_coordinates);
        touch_color = (TextView) findViewById(R.id.touch_color);
        mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.opencv_tutorial_activity_surface_view);
        mOpenCvCameraView.setVisibility(SurfaceView.VISIBLE);
        mOpenCvCameraView.setCvCameraViewListener(this);
    }

    @Override
    public void onPause() {
        super.onPause();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    @Override
    public void onResume()
    {
        super.onResume();
        if(!OpenCVLoader.initDebug())
        {
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_0_0,this,mLoaderCallback);
        }
        else
        {
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    }

    @Override
    public void onDestroy()
    {
        super.onDestroy();
        if (mOpenCvCameraView !=null)
            mOpenCvCameraView.disableView();
    }

    @Override
    public boolean onTouch(View view, MotionEvent motionEvent) {
        int cols=mRgba.cols();
        int rows =mRgba.rows();
        double yLow = (double)mOpenCvCameraView.getHeight()*0.2401961;
        double yHigh= (double)mOpenCvCameraView.getHeight()*0.7696078;
        double xScale = (double)cols/(double)mOpenCvCameraView.getWidth();
        double yscale=(double)rows/(yHigh-yLow);
        x = motionEvent.getX();
        y= motionEvent.getY();
        y = y-yLow;
        x=x*xScale;
        y=y*yscale;
        if((x<0)||(y<0) || (x>cols) || (y>rows)) return false;
        touch_coordinates.setText("x:" + Double.valueOf(x) + ",Y: " + Double.valueOf(y));
        Rect touchedRect = new Rect();

        touchedRect.x = (int)x;
        touchedRect.y = (int)y;

        touchedRect.width = 8;
        touchedRect.height= 8;

        Mat touchedRegionRgba =mRgba.submat(touchedRect);

        Mat touchedRegionHsv = new Mat();
        Imgproc.cvtColor(touchedRegionRgba, touchedRegionHsv, Imgproc.COLOR_RGB2HSV);

        mBlobColorHsv = Core.sumElems(touchedRegionHsv);
        int pointcount = touchedRect.width * touchedRect.height;
        for (int i=0; i< mBlobColorHsv.val.length; i++)
            mBlobColorHsv.val[i] /= pointcount;

        mBlobColorRgba = convertScalarHsv2Rgba(mBlobColorHsv);

        touch_color.setText("Color: #" + String.format("%02X", (int)mBlobColorRgba.val[0]) + String.format("%02X", (int)mBlobColorRgba.val[1]) +String.format("%02X", (int)mBlobColorRgba.val[2]));

        touch_color.setTextColor(Color.rgb((int) mBlobColorRgba.val[0], (int) mBlobColorRgba.val[1], (int) mBlobColorRgba.val[2]));
        touch_coordinates.setTextColor(Color.rgb((int) mBlobColorRgba.val[0], (int) mBlobColorRgba.val[1], (int) mBlobColorRgba.val[2]));

        image_base=mgray;
        detector.detect(image_base, keypoints1);
        descriptor.compute(image_base, keypoints1, descriptors1);
        activate=1;

        return false;
    }

    private Scalar convertScalarHsv2Rgba(Scalar hsvColor)
    {
        Mat pointMatRgba = new Mat();
        Mat pointMatHsv = new Mat(1,1, CvType.CV_8UC3, hsvColor);
        Imgproc.cvtColor(pointMatHsv, pointMatRgba, Imgproc.COLOR_HSV2RGB_FULL, 4);

        return new Scalar(pointMatRgba.get(0,0));
    }

    @Override
    public void onCameraViewStarted(int width, int height) {
        mRgba = new Mat();
        mgray = new Mat();
        image_base = new Mat();
        activate=0;

        descriptors1=new Mat();
        keypoints1=new MatOfKeyPoint();


        detector = FeatureDetector.create(FeatureDetector.ORB);
        descriptor = DescriptorExtractor.create(DescriptorExtractor.ORB);
        matcher = DescriptorMatcher.create(DescriptorMatcher.BRUTEFORCE_HAMMING);

        mBlobColorRgba = new Scalar(255);
        mBlobColorHsv = new Scalar(255);

    }

    @Override
    public void onCameraViewStopped() {
        mRgba.release();

    }

    private Mat recognize(Mat aInputFrame) {

        Mat descriptors2 = new Mat();
        MatOfKeyPoint keypoints2 = new MatOfKeyPoint();
        detector.detect(mgray, keypoints2);
        descriptor.compute(mgray, keypoints2, descriptors2);

        // Matching
        MatOfDMatch matches = new MatOfDMatch();
        if (mgray.type() == aInputFrame.type()) {
            matcher.match(descriptors1, descriptors2, matches);
        } else {
            return aInputFrame;
        }
        List<DMatch> matchesList = matches.toList();


        Double max_dist = 0.0;
        Double min_dist = 100.0;


        for (int i = 0; i < matchesList.size(); i++) {
            Double dist = (double) matchesList.get(i).distance;
            if (dist < min_dist)
                min_dist = dist;
            if (dist > max_dist)
                max_dist = dist;
        }

        LinkedList<DMatch> good_matches = new LinkedList<DMatch>();
        for (int i = 0; i < matchesList.size(); i++)  {
            if (matchesList.get(i).distance <= (3 * min_dist)) // change the limit as you desire
                good_matches.addLast(matchesList.get(i));
        }

        List<Point> objListGoodMatches = new ArrayList<Point>();
        List<Point> sceneListGoodMatches = new ArrayList<Point>();

        List<KeyPoint> keypoints_objectList = keypoints1.toList();
        List<KeyPoint> keypoints_sceneList = keypoints2.toList();

        for (int i = 0; i < good_matches.size(); i++) {
            // -- Get the keypoints from the good matches
            objListGoodMatches.add(keypoints_objectList.get(good_matches.get(i).queryIdx).pt);
            sceneListGoodMatches.add(keypoints_sceneList.get(good_matches.get(i).trainIdx).pt);
            circle(mgray, new Point(sceneListGoodMatches.get(i).x, sceneListGoodMatches.get(i).y), 3, new Scalar( 255, 0, 0, 255));

        }

        MatOfPoint2f objListGoodMatchesMat = new MatOfPoint2f();
        objListGoodMatchesMat.fromList(objListGoodMatches);
        MatOfPoint2f sceneListGoodMatchesMat = new MatOfPoint2f();
        sceneListGoodMatchesMat.fromList(sceneListGoodMatches);

        // findHomography needs 4 corresponding points
        if(good_matches.size()>3){

            Mat H = Calib3d.findHomography(objListGoodMatchesMat, sceneListGoodMatchesMat, Calib3d.RANSAC, 5 /* RansacTreshold */);

            Mat obj_corners = new Mat(4, 1, CvType.CV_32FC2);
            Mat scene_corners = new Mat(4, 1, CvType.CV_32FC2);

            obj_corners.put(0, 0, new double[] { 0, 0 });
            obj_corners.put(1, 0, new double[] { mgray.cols(), 0 });
            obj_corners.put(2, 0, new double[] { mgray.cols(), mgray.rows() });
            obj_corners.put(3, 0, new double[] { 0, mgray.rows() });

            Core.perspectiveTransform(obj_corners, scene_corners, H);

            line(mRgba, new Point(scene_corners.get(0, 0)), new Point(scene_corners.get(1, 0)), new Scalar(0, 255, 0), 2);
            line(mRgba, new Point(scene_corners.get(1, 0)), new Point(scene_corners.get(2, 0)), new Scalar(0, 255, 0), 2);
            line(mRgba, new Point(scene_corners.get(2, 0)), new Point(scene_corners.get(3, 0)), new Scalar(0, 255, 0), 2);
            line(mRgba, new Point(scene_corners.get(3, 0)), new Point(scene_corners.get(0, 0)), new Scalar(0, 255, 0), 2);

        }



        return mRgba;
    }

    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        mRgba=inputFrame.rgba();
        mgray = inputFrame.gray();
        Core.flip(mRgba.t(), mRgba, 1);
        Core.flip(mgray.t(), mgray, 1);

        if (activate==1) return recognize(mgray);

        return mRgba;
    }
}