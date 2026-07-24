#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    std::cout << "--- GAZE INTENT: C++ CORE INITIALIZED ---" << std::endl;

    // 0 represents the default system webcam
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "CRITICAL ERROR: C++ cannot open the webcam." << std::endl;
        return -1;
    }

    cv::Mat frame;
    while (true) {
        cap >> frame;
        if (frame.empty()) {
            std::cerr << "Dropped frame from buffer." << std::endl;
            break;
        }

        // Mirror the frame horizontally (matching the Python pipeline)
        cv::flip(frame, frame, 1);

        // Display the raw memory buffer
        cv::imshow("C++ Core Memory Buffer", frame);

        // Listen for the 'q' or 'Q' key to exit safely
        char key = (char)cv::waitKey(1);
        if (key == 'q' || key == 'Q') break;
    }

    cap.release();
    cv::destroyAllWindows();
    std::cout << "Engine safely shut down." << std::endl;
    return 0;
}