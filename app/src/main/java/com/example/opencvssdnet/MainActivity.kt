package com.example.opencvssdnet

import android.content.Context
import android.content.pm.PackageManager
import android.os.Bundle
import android.util.Log
import android.view.SurfaceView
import android.view.Window
import android.view.WindowManager
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import org.opencv.android.BaseLoaderCallback
import org.opencv.android.CameraBridgeViewBase
import org.opencv.android.LoaderCallbackInterface
import org.opencv.android.OpenCVLoader
import org.opencv.core.Mat
import org.opencv.core.Point
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.dnn.Dnn
import org.opencv.dnn.Net
import org.opencv.imgproc.Imgproc
import org.opencv.objdetect.CascadeClassifier
import java.io.BufferedInputStream
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.io.InputStream

class MainActivity : AppCompatActivity(), CameraBridgeViewBase.CvCameraViewListener2 {
    lateinit var cascFile: File
    var faceDetector: CascadeClassifier? = null
    lateinit var mOpenCvCameraView: CameraBridgeViewBase

    // Initialize OpenCV manager.
    private val mLoaderCallback: BaseLoaderCallback = object : BaseLoaderCallback(this) {
        override fun onManagerConnected(status: Int) {
            when (status) {
                SUCCESS -> {
                    Log.i(TAG, "OpenCV loaded successfully")
                    mOpenCvCameraView.enableView()
                    // Load the Cascade Classifier
                    val iStream: InputStream =
                        resources.openRawResource(R.raw.haarcascade_frontalface_alt2)
                    val cascadeDir = getDir("cascade", Context.MODE_PRIVATE)
                    cascFile = File(cascadeDir, "haarcasecade_frontalface_alt2.xml")
                    try {
                        val fos = FileOutputStream(cascFile)
                        val buffer = ByteArray(4096)
                        var bytesRead: Int
                        bytesRead = iStream.read(buffer)
                        while (bytesRead != -1) {
                            fos.write(buffer, 0, bytesRead)
                            bytesRead = iStream.read(buffer)
                        }
                        iStream.close()
                        fos.close()
                    } catch (e: IOException) {
                        e.printStackTrace()
                    }
                    faceDetector = CascadeClassifier(cascFile!!.absolutePath)
                    if (faceDetector!!.empty()) {
                        faceDetector = null
                    } else {
                        cascadeDir.delete()
                    }
                    mOpenCvCameraView.enableView()
                }

                else -> {
                    super.onManagerConnected(status)
                }
            }
        }
    }

    override fun onCameraViewStarted(width: Int, height: Int) {
        // Load a network
        val proto = getPath("MobileNetSSD_deploy.prototxt", this)
        val weights = getPath("MobileNetSSD_deploy.caffemodel", this)
        net = Dnn.readNetFromCaffe(proto, weights)
        Log.i(TAG, "Network loaded successfully")
    }

    // Upload file to storage and return a path.
    private fun getPath(file: String, context: Context): String {
        val assetManager = context.assets
        var inputStream: BufferedInputStream? = null
        try {
            // Read data from assets.
            inputStream = BufferedInputStream(assetManager.open(file))
            val data = ByteArray(inputStream.available())
            inputStream.read(data)
            inputStream.close()
            // Create copy file in storage.
            val outFile = File(context.filesDir, file)
            val os = FileOutputStream(outFile)
            os.write(data)
            os.close()
            // Return a path to file which may be read in common way.
            Log.i(TAG, "$file is successfully uploaded")
            return outFile.absolutePath
        } catch (ex: IOException) {
            Log.i(TAG, "Failed to upload a file")
        }
        return ""
    }

    override fun onCameraViewStopped() {}

    override fun onCameraFrame(inputFrame: CameraBridgeViewBase.CvCameraViewFrame): Mat? {
        // MobileNet Object Recognition
        val IN_WIDTH = 300
        val IN_HEIGHT = 300
        val WH_RATIO = IN_WIDTH.toFloat() / IN_HEIGHT.toFloat()
        val IN_SCALE_FACTOR = 0.007843
        val MEAN_VAL = 127.5
        val THRESHOLD = 0.2
        // Get a new frame
        val frame = inputFrame.rgba()
        Imgproc.cvtColor(frame, frame, Imgproc.COLOR_RGBA2RGB)
        // Forward image through network.
        val blob = Dnn.blobFromImage(
            frame,
            IN_SCALE_FACTOR,
            Size(IN_WIDTH.toDouble(), IN_HEIGHT.toDouble()),
            Scalar(MEAN_VAL, MEAN_VAL, MEAN_VAL), /*swapRB*/
            false, /*crop*/
            false,
        )
        net.setInput(blob)
        var detections = net.forward()
        val cols = frame.cols()
        val rows = frame.rows()
        detections = detections.reshape(1, detections.total().toInt() / 7)
        for (i in 0 until detections.rows()) {
            val confidence = detections[i, 2][0]
            if (confidence > THRESHOLD) {
                val classId = detections[i, 1][0].toInt()
                val left = (detections[i, 3][0] * cols).toInt()
                val top = (detections[i, 4][0] * rows).toInt()
                val right = (detections[i, 5][0] * cols).toInt()
                val bottom = (detections[i, 6][0] * rows).toInt()
                // Draw rectangle around detected object.
                Imgproc.rectangle(
                    frame,
                    Point(left.toDouble(), top.toDouble()),
                    Point(
                        right.toDouble(),
                        bottom.toDouble(),
                    ),
                    Scalar(0.0, 255.0, 0.0),
                )
                Log.i(TAG, "calassID" + classId + "confidence:" + confidence)
                val label = classNames[classId] + ": " + confidence
                val baseLine = IntArray(1)
                val labelSize =
                    Imgproc.getTextSize(label, Imgproc.FONT_HERSHEY_SIMPLEX, 0.5, 2, baseLine)
                // Draw background for label.
                Imgproc.rectangle(
                    frame,
                    Point(left.toDouble(), top - labelSize.height),
                    Point(left + labelSize.width, (top + baseLine[0]).toDouble()),
                    Scalar(255.0, 255.0, 255.0),
                    2,
                )
                // Write class name and confidence.
                Imgproc.putText(
                    frame,
                    label,
                    Point(left.toDouble(), top.toDouble()),
                    Imgproc.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    Scalar(0.0, 0.0, 0.0),
                )
            }
        }
        return frame
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        requestWindowFeature(Window.FEATURE_NO_TITLE)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        setContentView(R.layout.activity_main)
        if (ContextCompat.checkSelfPermission(
                this,
                android.Manifest.permission.CAMERA,
            ) != PackageManager.PERMISSION_GRANTED
        ) {
            ActivityCompat.requestPermissions(this, arrayOf(android.Manifest.permission.CAMERA), 0)
        }
        mOpenCvCameraView = findViewById(R.id.javaCameraView)
        mOpenCvCameraView.setCameraPermissionGranted()
        mOpenCvCameraView.visibility = SurfaceView.VISIBLE
        mOpenCvCameraView.setCvCameraViewListener(this)
    }

    public override fun onPause() {
        super.onPause()
        mOpenCvCameraView.disableView()
    }

    public override fun onResume() {
        super.onResume()
        if (!OpenCVLoader.initDebug()) {
            Log.d(TAG, "Internal OpenCV library not found. Using OpenCV Manager for initialization")
            OpenCVLoader.initAsync(
                OpenCVLoader.OPENCV_VERSION_3_0_0,
                this,
                mLoaderCallback,
            )
        } else {
            Log.d(TAG, "OpenCV library found inside package. Using it!")
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS)
        }
    }

    public override fun onDestroy() {
        super.onDestroy()
        mOpenCvCameraView.disableView()
    }

    companion object {
        private const val TAG = "MainActivity"
        private val classNames = arrayOf(
            "background",
            "aeroplane", "bicycle", "bird", "boat",
            "bottle", "bus", "car", "cat", "chair",
            "cow", "diningtable", "dog", "horse",
            "motorbike", "person", "pottedplant",
            "sheep", "sofa", "train", "tvmonitor",
        )
        lateinit var net: Net
    }
}
