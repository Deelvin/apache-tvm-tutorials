

package com.example.android.RVM.TVM

import android.Manifest
import android.annotation.SuppressLint
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Matrix
import android.os.Bundle
import android.util.Log
import android.util.Size
import android.view.SurfaceHolder
import android.view.View
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import com.android.example.RVM.TVM.databinding.ActivityCameraBinding
import java.io.IOException
import java.io.FileOutputStream
import java.io.InputStream
import java.io.File
import java.nio.charset.Charset
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit

import kotlin.random.Random

//import android.view.SurfaceHolder
/** Activity that displays the camera and performs object detection on the incoming frames */
class CameraActivity : AppCompatActivity(), SurfaceHolder.Callback {

    private lateinit var activityCameraBinding: ActivityCameraBinding

    private lateinit var resultBitmapBuffer: Bitmap

    private val executor = Executors.newSingleThreadExecutor()
    private val permissions = listOf(Manifest.permission.CAMERA)
    private val permissionsRequestCode = Random.nextInt(0, 10000)
//    private var lensFacing: Int = CameraSelector.LENS_FACING_BACK
    private var lensFacing: Int = CameraSelector.LENS_FACING_FRONT
    private val isFrontFacing get() = lensFacing == CameraSelector.LENS_FACING_FRONT

    private var pauseAnalysis = false
    private var imageRotationDegrees: Int = 0
    private var libFilename: String = "android.hg_demo_mobilenetv3.float16.atvm.720.so"
//    private var constsFilename: String = "consts_720_1280"
//    private var execFilename: String = "vm_exec_code_720_1280.ro"
      // LANDSCAPE layout!!
    private var TARGET_WIDTH = 1280
    private var TARGET_HEIGHT = 720
    private var fps_: Float = 0F
    val LOCK = Object()

    @Throws(IOException::class)
    private fun getTempLibFilePath(): String {
        val tempDir: File = File.createTempFile("tvm4j_demo_", "")
        if (!tempDir.delete() || !tempDir.mkdir()) {
            throw IOException("Couldn't create directory " + tempDir.getAbsolutePath())
        }

        return tempDir.path + File.separator.toString()
    }

    private fun copyToTempLib(pth: String, fileName: String) {
        val inputStream: InputStream = assets.open(fileName)
        val size: Int = inputStream.available()
        val buffer = ByteArray(size)
        inputStream.read(buffer)
        val fos = FileOutputStream(pth + fileName)
        fos.write(buffer)
        fos.close()
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        var array = getLibraryName()
        libFilename = String((array), Charset.defaultCharset())
        Log.d(TAG, "path is : ${libFilename}")
        var libCacheFilePath :String
        try {
            libCacheFilePath = getTempLibFilePath()
            copyToTempLib(libCacheFilePath, libFilename)
//            copyToTempLib(libCacheFilePath, constsFilename)
//            copyToTempLib(libCacheFilePath, execFilename)
            Log.d(TAG, "path is : ${libCacheFilePath}")
        } catch (e: IOException) {
            Log.e(TAG, "Problem uploading compiled function!", e)
            return  //failure
        }

        activityCameraBinding = ActivityCameraBinding.inflate(layoutInflater)
        setContentView(activityCameraBinding.root)
        val res = initRVM(libCacheFilePath)

        Log.d(TAG, "Init result: ${res}")
        activityCameraBinding.cameraCaptureButton.setOnClickListener {

            // Disable all camera controls
            it.isEnabled = false

            if (pauseAnalysis) {
//                // If image analysis is in paused state, resume it
//                pauseAnalysis = false
//                activityCameraBinding.imagePredicted.visibility = View.VISIBLE
//                activityCameraBinding.imagePredicted.setImageBitmap(bitmapBuffer)
//            } else {
//                // Otherwise, pause image analysis and freeze image
//                pauseAnalysis = true
//                val matrix = Matrix().apply {
//                    postRotate(imageRotationDegrees.toFloat())
//                    if (isFrontFacing) postScale(-1f, 1f)
//                }
//                val uprightImage = Bitmap.createBitmap(
//                    bitmapBuffer, 0, 0, bitmapBuffer.width, bitmapBuffer.height, matrix, true)
//                activityCameraBinding.imagePredicted.setImageBitmap(uprightImage)
//                activityCameraBinding.imagePredicted.visibility = View.VISIBLE
//                activityCameraBinding.viewFinder.visibility = View.GONE
            }

            // Re-enable camera controls
            it.isEnabled = true
        }
    }

    override fun surfaceCreated(surfaceHolder: SurfaceHolder) {
        Log.d(TAG, "Created: ")
    }
    override fun surfaceChanged(surfaceHolder: SurfaceHolder, i: Int, i1: Int, i2: Int) {
        Log.d(TAG, "Changed: ")
    }

    override fun surfaceDestroyed(surfaceHolder: SurfaceHolder) {
        Log.d(TAG, "Destroyed: ")
    }

    override fun onDestroy() {

        // Terminate all outstanding analyzing jobs (if there is any).
        executor.apply {
            shutdown()
            awaitTermination(1000, TimeUnit.MILLISECONDS)
        }

        releaseRVM()
        super.onDestroy()
    }

    /** Declare and bind preview and analysis use cases */
    @SuppressLint("UnsafeExperimentalUsageError")
    private fun bindCameraUseCases() = activityCameraBinding.viewFinder.post {

        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener ({
            // Camera provider is now guaranteed to be available
            val cameraProvider = cameraProviderFuture.get()

            val preview = Preview.Builder()
                .setTargetAspectRatio(AspectRatio.RATIO_16_9)
                .setTargetRotation(activityCameraBinding.viewFinder.display.rotation)
                .build()

            // Set up the image analysis use case which will process frames in real time
            val imageAnalysis = ImageAnalysis.Builder()
                .setTargetResolution(Size(TARGET_WIDTH, TARGET_HEIGHT))
                .setTargetRotation(activityCameraBinding.viewFinder.display.rotation)
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build()

            var frameCounter = 0
            var lastFpsTimestamp = System.currentTimeMillis()

            imageAnalysis.setAnalyzer(executor, ImageAnalysis.Analyzer { image ->
                if (!::resultBitmapBuffer.isInitialized) {
                    // The image rotation and RGB image buffer are initialized only once
                    // the analyzer has started running
                    imageRotationDegrees = image.imageInfo.rotationDegrees
                    activityCameraBinding.viewFinder.visibility = View.INVISIBLE
                    activityCameraBinding.imagePredicted.visibility = View.VISIBLE
                    resultBitmapBuffer = Bitmap.createBitmap(
                        image.width, image.height, Bitmap.Config.ARGB_8888)
                }

                // Copy out RGB bits to our shared buffer
                var status = -1;
                image.use {
                    synchronized(LOCK) {
                        resultBitmapBuffer.copyPixelsFromBuffer(image.planes[0].buffer)
                        status = updateBitmap(resultBitmapBuffer, image)
                    }
                }
                if (status == 0){
                    drawResult()
                }

                // Compute the FPS of the entire pipeline
                val frameCount = 10
                if (++frameCounter % frameCount == 0) {
                    frameCounter = 0
                    val now = System.currentTimeMillis()
                    val delta = now - lastFpsTimestamp
                    fps_ = 1000 * frameCount.toFloat() / delta
                    Log.d(TAG, "FPS: ${"%.02f".format(fps_)} with image: ${image.width} x ${image.height}")
                    lastFpsTimestamp = now
                }
            })

            // Create a new camera selector each time, enforcing lens facing
            val cameraSelector = CameraSelector.Builder().requireLensFacing(lensFacing).build()

            // Apply declared configs to CameraX using the same lifecycle owner
            cameraProvider.unbindAll()
            cameraProvider.bindToLifecycle(
                this as LifecycleOwner, cameraSelector, preview, imageAnalysis)
            preview.setSurfaceProvider(activityCameraBinding.viewFinder.surfaceProvider)
        }, ContextCompat.getMainExecutor(this))
    }

    private fun drawResult() = activityCameraBinding.viewFinder.post {
        val matrix = Matrix().apply {
            postRotate(imageRotationDegrees.toFloat())
            if (isFrontFacing) postScale(-1f, 1f)
        }
        synchronized(LOCK)
        {
            val uprightImage = Bitmap.createBitmap(
                resultBitmapBuffer, 0, 0, resultBitmapBuffer.width,
                resultBitmapBuffer.height, matrix, true)
            activityCameraBinding.imagePredicted.setImageBitmap(uprightImage)
        }
        activityCameraBinding.textPrediction.text = "FPS: ${"%.2f".format(fps_)}"
    }

    override fun onResume() {
        super.onResume()

        // Request permissions each time the app resumes, since they can be revoked at any time
        if (!hasPermissions(this)) {
            ActivityCompat.requestPermissions(
                this, permissions.toTypedArray(), permissionsRequestCode)
        } else {
            bindCameraUseCases()
        }
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == permissionsRequestCode && hasPermissions(this)) {
            bindCameraUseCases()
        } else {
            finish() // If we don't have the required permissions, we can't run
        }
    }

    /** Convenience method used to check if all permissions required by this app are granted */
    private fun hasPermissions(context: Context) = permissions.all {
        ContextCompat.checkSelfPermission(context, it) == PackageManager.PERMISSION_GRANTED
    }
    external fun initRVM(path: String) : Int;
    external fun releaseRVM() : Int;
    external fun updateBitmap(image : Bitmap, proxy: ImageProxy) : Int;
    external fun clearRNNState();
    external fun getLibraryName() :ByteArray
    companion object {
        init {
            System.loadLibrary("tvm_rvm")
        }

        private val TAG = CameraActivity::class.java.simpleName
    }
}
