#include <iostream>
#include <fstream>
#include <sstream>
#include <unistd.h>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#ifdef CV_CXX11
#include <mutex>
#include <thread>
#include <queue>
#endif

#include "common.hpp"

std::string keys =
    "{ help  h     | | Print help message. }"
    "{ @alias      | | An alias name of model to extract preprocessing parameters from models.yml file. }"
    "{ zoo         | models.yml | An optional path to file with preprocessing parameters }"
    "{ device      |  0 | camera device number. }"
    "{ input i     | | Path to input image or video file. Skip this argument to capture frames from a camera. }"
    "{ framework f | | Optional name of an origin framework of the model. Detect it automatically if it does not set. }"
    "{ classes     | | Optional path to a text file with names of classes to label detected objects. }"
    "{ thr         | .50 | Confidence threshold. }"
    "{ nms         | .4 | Non-maximum suppression threshold. }"
    "{ backend     |  0 | Choose one of computation backends: "
    "0: automatically (by default), "
    "1: Halide language (http://halide-lang.org/), "
    "2: Intel's Deep Learning Inference Engine (https://software.intel.com/openvino-toolkit), "
    "3: OpenCV implementation }"
    "{ target      | 0 | Choose one of target computation devices: "
    "0: CPU target (by default), "
    "1: OpenCL, "
    "2: OpenCL fp16 (half-float precision), "
    "3: VPU }"
    "{ async       | 0 | Number of asynchronous forwards at the same time. "
    "Choose 0 for synchronous mode }";

using namespace cv;
using namespace dnn;

float confThreshold, nmsThreshold;
std::vector<std::string> classes;

inline void preprocess(const Mat &frame, Net &net, Size inpSize, float scale,
                       const Scalar &mean, bool swapRB);

void postprocess(Mat &frame, const std::vector<Mat> &out, Net &net);

#ifdef CV_CXX11
template <typename T>
class QueueFPS : public std::queue<T>
{
public:
    QueueFPS() : counter(0) {}

    void push(const T &entry)
    {
        std::lock_guard<std::mutex> lock(mutex);

        std::queue<T>::push(entry);
        counter += 1;
    }

    T get()
    {
        std::lock_guard<std::mutex> lock(mutex);
        T entry = this->front();
        this->pop();
        return entry;
    }

    void clear()
    {
        std::lock_guard<std::mutex> lock(mutex);
        while (!this->empty())
            this->pop();
    }

    unsigned int counter;

private:
    std::mutex mutex;
};
#endif // CV_CXX11

int main(int argc, char **argv)
{
    char cmdline[256];
    sprintf(cmdline, "echo %d > /sys/fs/cgroup/memory/OPENCVDNN/cgroup.procs", getpid());
    system(cmdline);

    CommandLineParser parser(argc, argv, keys);

    const std::string modelName = parser.get<String>("@alias");
    const std::string zooFile = parser.get<String>("zoo");

    keys += genPreprocArguments(modelName, zooFile);

    parser = CommandLineParser(argc, argv, keys);
    parser.about("Use this script to run object detection deep learning networks using OpenCV.");
    if (argc == 1 || parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }

    confThreshold = parser.get<float>("thr");
    nmsThreshold = parser.get<float>("nms");

    float scale = parser.get<float>("scale");
    Scalar mean = parser.get<Scalar>("mean");
    bool swapRB = parser.get<bool>("rgb");
    int inpWidth = parser.get<int>("width");
    int inpHeight = parser.get<int>("height");

    CV_Assert(parser.has("model"));

    std::string modelPath = findFile(parser.get<String>("model"));
    std::string configPath = findFile(parser.get<String>("config"));

    // Open file with classes names.
    if (parser.has("classes"))
    {
        std::string file = parser.get<String>("classes");
        std::ifstream ifs(file.c_str());
        if (!ifs.is_open())
            CV_Error(Error::StsError, "File " + file + " not found");
        std::string line;
        while (std::getline(ifs, line))
        {
            classes.push_back(line);
        }
    }

    // Load a model.
    Net net = readNetFromDarknet(configPath, modelPath);
    //Net net = readNet(modelPath, configPath, parser.get<String>("framework"));
    net.setPreferableBackend(parser.get<int>("backend")); // Automatically
    net.setPreferableTarget(parser.get<int>("target"));   // CPU
    std::vector<String> outNames = net.getUnconnectedOutLayersNames();

    // Open a video file or an image file or a camera stream.
    VideoCapture cap;
    if (parser.has("input"))
        cap.open(parser.get<String>("input"));
    else
        std::exit(EXIT_FAILURE); // no webcam support

    // Sequential read of frames
    int frame_count = 1;
    Mat frame;
    std::vector<Mat> outs;
    std::vector<String> netLayerNames;

    //Start of the program

    // initial frame
    cap >> frame;
    while (!frame.empty())
    {
        preprocess(frame, net, Size(inpWidth, inpHeight), scale, mean, swapRB);
#ifdef NEVER
        netLayerNames = net.getLayerNames();

        Mat output;

        for (auto layer : netLayerNames) {
            std::cout << layer << std::endl;
            output = net.forwardSingleLayer(layer);
        }
#else
        net.forward(outs, outNames);
#endif
        std::cout << "\nFrame : " << frame_count++ << std::endl;
        postprocess(frame, outs, net);

        cap >> frame;
    }

    return EXIT_SUCCESS;
}

inline void preprocess(const Mat &frame, Net &net, Size inpSize, float scale,
                       const Scalar &mean, bool swapRB)
{
    static Mat blob;
    // Create a 4D blob from a frame.
    if (inpSize.width <= 0)
        inpSize.width = frame.cols;
    if (inpSize.height <= 0)
        inpSize.height = frame.rows;
    blobFromImage(frame, blob, 1.0, inpSize, Scalar(), swapRB, false, CV_8U);

    // Run a model.
    net.setInput(blob, "", scale, mean);
    if (net.getLayer(0)->outputNameToIndex("im_info") != -1) // Faster-RCNN or R-FCN
    {
        resize(frame, frame, inpSize);
        Mat imInfo = (Mat_<float>(1, 3) << inpSize.height, inpSize.width, 1.6f);
        net.setInput(imInfo, "im_info");
    }
}

void postprocess(Mat &frame, const std::vector<Mat> &outs, Net &net)
{
    static std::vector<int> outLayers = net.getUnconnectedOutLayers();
    static std::string outLayerType = net.getLayer(outLayers[0])->type;
    std::string output;

    if (outLayerType == "DetectionOutput")
    {
        // Network produces output blob with a shape 1x1xNx7 where N is a number of
        // detections and an every detection is a vector of values
        // [batchId, classId, confidence, left, top, right, bottom]
        CV_Assert(outs.size() > 0);
        for (size_t k = 0; k < outs.size(); k++)
        {
            float *data = (float *)outs[k].data;
            for (size_t i = 0; i < outs[k].total(); i += 7)
            {
                float confidence = data[i + 2];
                if (confidence > confThreshold)
                {
                    int classId = ((int)(data[i + 1]) - 1);
                    if (!classes.empty())
                    {
                        CV_Assert(classId < (int)classes.size());
                        output = classes[classId] + "\t -> confidence : ";
                        std::cout << output << confidence << std::endl;
                    }
                }
            }
        }
    }
    else if (outLayerType == "Region")
    {
        for (size_t i = 0; i < outs.size(); ++i)
        {
            // Network produces output blob with a shape NxC where N is a number of
            // detected objects and C is a number of classes + 4 where the first 4
            // numbers are [center_x, center_y, width, height]
            float *data = (float *)outs[i].data;
            for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols)
            {
                Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
                Point classIdPoint;
                double confidence;
                minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);

                //TODO: this could be inlined and the above also
                if (confidence > confThreshold)
                {
                    int classId = classIdPoint.x;
                    if (!classes.empty())
                    {
                        CV_Assert(classId < (int)classes.size());
                        output = classes[classId] + "\t -> confidence : ";
                        std::cout << output << confidence << std::endl;
                    }
                }
            }
        }
    }
    else
        CV_Error(Error::StsNotImplemented, "Unknown output layer type: " + outLayerType);
}