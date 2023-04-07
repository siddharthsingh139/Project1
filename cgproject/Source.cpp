#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

using namespace cv;
using namespace std;

double getEnergy(Mat mat, int i, int j, int size) {
    if (j < 0) {
        return mat.at<double>(i, 0);
    }
    if (j >= size) {
        return mat.at<double>(i, size - 1);
    }
    return mat.at<double>(i, j);
}

// https://docs.opencv.org/3.4/d2/d2c/tutorial_sobel_derivatives.html
Mat generateEnergyMap(Mat& img) {
    // Gaussian Blur to reduceWidth noise
    Mat blurImage;
    GaussianBlur(img, blurImage, Size(3, 3), 0, 0, BORDER_DEFAULT);

    // Grayscale
    Mat grayImage;
    cvtColor(blurImage, grayImage, COLOR_BGR2GRAY);

    // Sobel operator to compute the gradient of the image
    Mat xGradient, yGradient;
    Sobel(grayImage, xGradient, CV_16S, 1, 0, 3, 1, 0, BORDER_DEFAULT);
    Sobel(grayImage, yGradient, CV_16S, 0, 1, 3, 1, 0, BORDER_DEFAULT);

    // Convert gradients to absolute
    Mat xGradAbs, yGradAbs;
    convertScaleAbs(xGradient, xGradAbs);
    convertScaleAbs(yGradient, yGradAbs);

    // Summation of gradiants
    Mat gradient;
    addWeighted(xGradAbs, 0.5, yGradAbs, 0.5, 0, gradient);

    // Convert to double
    Mat energyImage;
    gradient.convertTo(energyImage, CV_64F, 1.0 / 255.0);

    //imshow("Energy Map", energyImage);

    return energyImage;
}

Mat generateCumulativeMap(Mat& eImg, Size_<int> size) {
    Mat cMap = Mat(size, CV_64F);

    // Base Condition
    for (int i = 0; i < size.width; ++i) {
        cMap.at<double>(0, i) = eImg.at<double>(0, i);
    }

    // Dynamic Programming
    for (int i = 1; i < eImg.rows; i++) {
        for (int j = 0; j < eImg.cols; j++) {
            cMap.at<double>(i, j) = eImg.at<double>(i, j) +
                min({
                    getEnergy(cMap, i - 1, j - 1, size.width),
                    getEnergy(cMap, i - 1, j, size.width),
                    getEnergy(cMap, i - 1, j + 1, size.width)
            });
        }
    }

    return cMap;
}

vector<int> findSeam(Mat& cMap, Size_<int> size) {
    vector<int> optPath(size.height);
    double Min = 1e9;
    int j = -1; // Min Index ie index at which Min val occurs

    for (int i = 0; i < size.width; ++i) {
        double val = cMap.at<double>(size.height - 1, i);
        if (val < Min) {
            Min = val;
            j = i;
        }
    }
    optPath[size.height - 1] = j;

    for (int i = size.height - 2; i >= 0; i--) {
        double a = getEnergy(cMap, i, j - 1, size.width),
            b = getEnergy(cMap, i, j, size.width),
            c = getEnergy(cMap, i, j + 1, size.width);

        if (min({ a, b, c }) == c) {
            j++;
        }
        else if (min({ a, b, c }) == a) {
            j--;
        }
        if (j < 0) j = 0;
        else if (j >= size.width) j = size.width - 1;

        optPath[i] = j;
    }

    return optPath;
}

void reduceWidth(Mat& img, vector<int> path, Size_<int> size) {
    for (int i = 0; i < size.height; i++) {
        int k = 0;
        for (int j = 0; j < size.width; ++j) {
            if (j == path[i]) continue;
            img.at<Vec3b>(i, k++) = img.at<Vec3b>(i, j);
        }
    }
    img = img.colRange(0, size.width - 1);

    imshow("Reduced Image", img);
}

int main() {
    string filename = "original.jpg";

    Mat image = imread(filename);
    if (image.empty()) {
        cout << "Image not found\n";
        return 0;
    }

    Size_<int> imgSize = Size(image.cols, image.rows);

    imshow("Original Image", image);

    for (int i = 0; i < 200; i++) {
        Mat energyMap = generateEnergyMap(image);
        Mat cumulativeMap = generateCumulativeMap(energyMap, imgSize);
        vector<int> path = findSeam(cumulativeMap, imgSize);

        for (int j = 0; j < energyMap.rows; j++) {
            energyMap.at<double>(j, path[j]) = 1;
        }
        imshow("Seam Path", energyMap);

        waitKey(10);
        reduceWidth(image, path, imgSize);
        imgSize.width--;
    }
    waitKey(0);
    return 0;
}