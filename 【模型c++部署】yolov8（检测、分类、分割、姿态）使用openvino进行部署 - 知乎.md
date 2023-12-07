# 【模型c++部署】yolov8（检测、分类、分割、姿态）使用openvino进行部署

[![努力的袁](https://pic1.zhimg.com/v2-83582cefa72468b231bad25dca0b4be7_l.jpg?source=172ae18b)](https://zhuanlan.zhihu.com//www.zhihu.com/people/yuan-jiang-jiang-jiang)

[努力的袁](https://zhuanlan.zhihu.com//www.zhihu.com/people/yuan-jiang-jiang-jiang)

系外行星的研究生

> 该文主要是对yolov8的检测、分类、分割、姿态应用使用c++进行dll封装，并进行调用测试。

## 0\. 模型准备

openvino调用的是xml和bin文件（下面的推理方式只需要调用xml的文件就行，另外一篇（[链接](https://link.zhihu.com/?target=https%3A//blog.csdn.net/qq_44747572/article/details/134003776%3Fspm%3D1001.2014.3001.5501)）使用xml和bin文件调用的）。 文件的获取过程（yolov8是pytorch训练的）： **pt->onnx->openvino（xml和bin）**  
openvino可调的是xml和bin文件[](https://link.zhihu.com/?target=https%3A//blog.csdn.net/qq_44747572/article/details/134003776%3Fspm%3D1001.2014.3001.5501)文件的获取过程**\>>**

### 方法一：（使用这种，由于版本不一致，推理失败）

使用yolov8自带的代码进行转换，这个过程比较方便，但是对于后续部署其他的模型不太方便。

```cpp
path = model.export(format="openvino")这行代码可以直接将yolov8n-pose.pt模型转换为xml和bin文件
# 加载预训练模型
    model = YOLO("yolov8n-pose.pt") 
    #path = model.export(format="onnx")
    path = model.export(format="openvino")
    # model = YOLO("yolov8n.pt") task参数也可以不填写，它会根据模型去识别相应任务类别
    # 检测图片
    results = model("./ultralytics/assets/bus.jpg")
    res = results[0].plot()
    cv2.imshow("YOLOv8 Inference", res)
    cv2.waitKey(0)
```

  

![](https://pic4.zhimg.com/v2-6834398a16ddd42ae281a3f1a2f532ab_b.jpg)

![](https://pic4.zhimg.com/80/v2-6834398a16ddd42ae281a3f1a2f532ab_720w.webp)

在这里插入图片描述

  

### 方法二：

1.  使用python的环境进行配置：[pip下载方法](https://link.zhihu.com/?target=https%3A//www.intel.cn/content/www/cn/zh/developer/tools/openvino-toolkit/download.html%3FVERSION%3Dv_2023_1_0%26OP_SYSTEM%3DWINDOWS%26DISTRIBUTION%3DPIP)  
    使用Python环境进行配置[](https://link.zhihu.com/?target=https%3A//www.intel.cn/content/www/cn/zh/developer/tools/openvino-toolkit/download.html%3FVERSION%3Dv_2023_1_0%26OP_SYSTEM%3DWINDOWS%26DISTRIBUTION%3DPIP)

![](https://pic4.zhimg.com/v2-ac20f923504f97a0a0243804579d367b_b.jpg)

![](https://pic4.zhimg.com/80/v2-ac20f923504f97a0a0243804579d367b_720w.webp)

在这里插入图片描述

![](https://pic4.zhimg.com/v2-b2fe90bed77301c48e3508115635b293_b.jpg)

![](https://pic4.zhimg.com/80/v2-b2fe90bed77301c48e3508115635b293_720w.webp)

在这里插入图片描述

主要就是：`pip install openvino==2023.1.0`

1.  使用代码行进行推理 在对应的环境库中找到mo\_onnx.py，在终端切换路径到mo\_onnx.py的路径下，然后再使用下面的命令。

```cpp
python mo_onnx.py --input_model D:\Users\6536\Desktop\python\onnx2openvino\yolov8n.onnx --output_dir D:\Users\6536\Desktop\python\onnx2openvino
```

### 方法三：（推荐这种）

[https://zhuanlan.zhihu.com/p/358437476](https://zhuanlan.zhihu.com/p/358437476)

## 1\. OV\_YOLOV8\_DLL

  

![](https://pic2.zhimg.com/v2-6ac377d205a511990af1afa550b2f8f1_b.jpg)

![](https://pic2.zhimg.com/80/v2-6ac377d205a511990af1afa550b2f8f1_720w.webp)

在这里插入图片描述

  

### 0\. c++依赖项配置

主要配置opencv以及openvino openvino的配置：[链接](https://link.zhihu.com/?target=https%3A//blog.csdn.net/qq_44747572/article/details/134003776%3Fspm%3D1001.2014.3001.5501) 所以配置截图：  
主要配置opencv和openvino openvino的配置：[](https://link.zhihu.com/?target=https%3A//blog.csdn.net/qq_44747572/article/details/134003776%3Fspm%3D1001.2014.3001.5501)

![](https://pic2.zhimg.com/v2-b6af8b513c8c07ce759d66a9e9e5b461_b.jpg)

![](https://pic2.zhimg.com/80/v2-b6af8b513c8c07ce759d66a9e9e5b461_720w.webp)

在这里插入图片描述

![](https://pic1.zhimg.com/v2-2a71c5a170769e32823bad401cfe1490_b.jpg)

![](https://pic1.zhimg.com/80/v2-2a71c5a170769e32823bad401cfe1490_720w.webp)

在这里插入图片描述

  

  

![](https://pic1.zhimg.com/v2-bc916a159810b91441618235db7afda0_b.jpg)

![](https://pic1.zhimg.com/80/v2-bc916a159810b91441618235db7afda0_720w.webp)

在这里插入图片描述

![](https://pic4.zhimg.com/v2-3a3bc9be8a0a24a25d112f63165297cf_b.jpg)

![](https://pic4.zhimg.com/80/v2-3a3bc9be8a0a24a25d112f63165297cf_720w.webp)

在这里插入图片描述

  

### 1\. ov\_yolov8.cpp

```cpp
#include "ov_yolov8.h"


// 全局变量
std::vector<cv::Scalar> colors = { cv::Scalar(0, 0, 255) , cv::Scalar(0, 255, 0) , cv::Scalar(255, 0, 0) ,
                               cv::Scalar(255, 100, 50) , cv::Scalar(50, 100, 255) , cv::Scalar(255, 50, 100) };


std::vector<Scalar> colors_seg = { Scalar(255, 0, 0), Scalar(255, 0, 255), Scalar(170, 0, 255), Scalar(255, 0, 85),
                                   Scalar(255, 0, 170), Scalar(85, 255, 0), Scalar(255, 170, 0), Scalar(0, 255, 0),
                                   Scalar(255, 255, 0), Scalar(0, 255, 85), Scalar(170, 255, 0), Scalar(0, 85, 255),
                                   Scalar(0, 255, 170), Scalar(0, 0, 255), Scalar(0, 255, 255), Scalar(85, 0, 255) };

// 定义skeleton的连接关系以及color mappings
std::vector<std::vector<int>> skeleton = { {16, 14}, {14, 12}, {17, 15}, {15, 13}, {12, 13}, {6, 12}, {7, 13}, {6, 7},
                                          {6, 8}, {7, 9}, {8, 10}, {9, 11}, {2, 3}, {1, 2}, {1, 3}, {2, 4}, {3, 5}, {4, 6}, {5, 7} };

std::vector<cv::Scalar> posePalette = {
        cv::Scalar(255, 128, 0), cv::Scalar(255, 153, 51), cv::Scalar(255, 178, 102), cv::Scalar(230, 230, 0), cv::Scalar(255, 153, 255),
        cv::Scalar(153, 204, 255), cv::Scalar(255, 102, 255), cv::Scalar(255, 51, 255), cv::Scalar(102, 178, 255), cv::Scalar(51, 153, 255),
        cv::Scalar(255, 153, 153), cv::Scalar(255, 102, 102), cv::Scalar(255, 51, 51), cv::Scalar(153, 255, 153), cv::Scalar(102, 255, 102),
        cv::Scalar(51, 255, 51), cv::Scalar(0, 255, 0), cv::Scalar(0, 0, 255), cv::Scalar(255, 0, 0), cv::Scalar(255, 255, 255)
};

std::vector<int> limbColorIndices = { 9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16 };
std::vector<int> kptColorIndices = { 16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9 };


const std::vector<std::string> class_names = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush" };



YoloModel::YoloModel()
{

}
YoloModel::~YoloModel()
{

}

// =====================检测========================//
bool YoloModel::LoadDetectModel(const string& xmlName, string& device)
{
    //待优化，如何把初始化部分进行提取出来
   // -------- Step 1. Initialize OpenVINO Runtime Core --------
    ov::Core core;

    // -------- Step 2. Compile the Model --------
    compiled_model_Detect = core.compile_model(xmlName, device);

    // -------- Step 3. Create an Inference Request --------
    infer_request_Detect = compiled_model_Detect.create_infer_request();

    return true;
}


bool YoloModel::YoloDetectInfer(const Mat& src, double cof_threshold, double nms_area_threshold, Mat& dst, vector<Object>& vecObj)
{
    // -------- Step 4.Read a picture file and do the preprocess --------
    // Preprocess the image
    Mat letterbox_img;
    letterbox(src, letterbox_img);
    float scale = letterbox_img.size[0] / 640.0;
    Mat blob = blobFromImage(letterbox_img, 1.0 / 255.0, Size(640, 640), Scalar(), true);

    // -------- Step 5. Feed the blob into the input node of the Model -------
    // Get input port for model with one input
    auto input_port = compiled_model_Detect.input();
    // Create tensor from external memory
    ov::Tensor input_tensor(input_port.get_element_type(), input_port.get_shape(), blob.ptr(0));
    // Set input tensor for model with one input
    infer_request_Detect.set_input_tensor(input_tensor);
    // -------- Step 6. Start inference --------
    infer_request_Detect.infer();

    // -------- Step 7. Get the inference result --------
    auto output = infer_request_Detect.get_output_tensor(0);
    auto output_shape = output.get_shape();
    std::cout << "The shape of output tensor:" << output_shape << std::endl;
    int rows = output_shape[2];        //8400
    int dimensions = output_shape[1];  //84: box[cx, cy, w, h]+80 classes scores



    // -------- Step 8. Postprocess the result --------
    float* data = output.data<float>();
    Mat output_buffer(output_shape[1], output_shape[2], CV_32F, data);
    transpose(output_buffer, output_buffer); //[8400,84]
    float score_threshold = 0.25;
    float nms_threshold = 0.5;
    std::vector<int> class_ids;
    std::vector<float> class_scores;
    std::vector<Rect> boxes;

    // Figure out the bbox, class_id and class_score
    for (int i = 0; i < output_buffer.rows; i++) {
        Mat classes_scores = output_buffer.row(i).colRange(4, 84);
        Point class_id;
        double maxClassScore;
        minMaxLoc(classes_scores, 0, &maxClassScore, 0, &class_id);

        if (maxClassScore > score_threshold) {
            class_scores.push_back(maxClassScore);
            class_ids.push_back(class_id.x);
            float cx = output_buffer.at<float>(i, 0);
            float cy = output_buffer.at<float>(i, 1);
            float w = output_buffer.at<float>(i, 2);
            float h = output_buffer.at<float>(i, 3);

            int left = int((cx - 0.5 * w) * scale);
            int top = int((cy - 0.5 * h) * scale);
            int width = int(w * scale);
            int height = int(h * scale);

            boxes.push_back(Rect(left, top, width, height));
        }
    }
    //NMS
    std::vector<int> indices;
    NMSBoxes(boxes, class_scores, score_threshold, nms_threshold, indices);

    // -------- Visualize the detection results -----------
    dst = src.clone();
    for (size_t i = 0; i < indices.size(); i++) {
        int index = indices[i];
        int class_id = class_ids[index];
        rectangle(dst, boxes[index], colors[class_id % 6], 2, 8);
        std::string label = class_names[class_id] + ":" + std::to_string(class_scores[index]).substr(0, 4);
        Size textSize = cv::getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, 0);
        Rect textBox(boxes[index].tl().x, boxes[index].tl().y - 15, textSize.width, textSize.height + 5);
        cv::rectangle(dst, textBox, colors[class_id % 6], FILLED);
        putText(dst, label, Point(boxes[index].tl().x, boxes[index].tl().y - 5), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255));
    }


    return true;
}


// =====================分类========================//
bool YoloModel::LoadClsModel(const string& xmlName, string& device)
{
    //待优化，如何把初始化部分进行提取出来
   // -------- Step 1. Initialize OpenVINO Runtime Core --------
    ov::Core core;

    // -------- Step 2. Compile the Model --------
    compiled_model_Detect_Cls = core.compile_model(xmlName, device);

    // -------- Step 3. Create an Inference Request --------
    infer_request_Cls = compiled_model_Detect_Cls.create_infer_request();

    return true;
}




bool YoloModel::YoloClsInfer(const Mat& src, double cof_threshold, double nms_area_threshold, Mat& dst, vector<Object>& vecObj)
{
    // -------- Step 4.Read a picture file and do the preprocess --------
    // Preprocess the image
    Mat letterbox_img;
    letterbox(src, letterbox_img);
    float scale = letterbox_img.size[0] / 640.0;
    Mat blob = blobFromImage(letterbox_img, 1.0 / 255.0, Size(224, 224), Scalar(), true);

    // -------- Step 5. Feed the blob into the input node of the Model -------
    // Get input port for model with one input
    auto input_port = compiled_model_Detect_Cls.input();
    // Create tensor from external memory
    ov::Tensor input_tensor(input_port.get_element_type(), input_port.get_shape(), blob.ptr(0));
    // Set input tensor for model with one input
    infer_request_Cls.set_input_tensor(input_tensor);

    // -------- Step 6. Start inference --------
    infer_request_Cls.infer();

    // -------- Step 7. Get the inference result --------
    auto output = infer_request_Cls.get_output_tensor(0);
    auto output_shape = output.get_shape();
    std::cout << "The shape of output tensor:" << output_shape << std::endl;

    // -------- Step 8. Postprocess the result --------
    float* output_buffer = output.data<float>();
    std::vector<float> result(output_buffer, output_buffer + output_shape[1]);
    auto max_idx = std::max_element(result.begin(), result.end());
    int class_id = max_idx - result.begin();
    float score = *max_idx;
    std::cout << "Class ID:" << class_id << " Score:" << score << std::endl;

    return true;
}


// =====================分割========================//
bool YoloModel::LoadSegModel(const string& xmlName, string& device)
{
    //待优化，如何把初始化部分进行提取出来
   // -------- Step 1. Initialize OpenVINO Runtime Core --------
    ov::Core core;

    // -------- Step 2. Compile the Model --------
    compiled_model_Seg = core.compile_model(xmlName, device);

    // -------- Step 3. Create an Inference Request --------
    infer_request_Seg = compiled_model_Seg.create_infer_request();

    return true;
}




bool YoloModel::YoloSegInfer(const Mat& src, double cof_threshold, double nms_area_threshold, Mat& dst, vector<Object>& vecObj)
{
    // -------- Step 4.Read a picture file and do the preprocess --------
    // Preprocess the image
    Mat letterbox_img;
    letterbox(src, letterbox_img);
    float scale = letterbox_img.size[0] / 640.0;
    Mat blob = blobFromImage(letterbox_img, 1.0 / 255.0, Size(640, 640), Scalar(), true);

    // -------- Step 5. Feed the blob into the input node of the Model -------
    // Get input port for model with one input
    auto input_port = compiled_model_Seg.input();
    // Create tensor from external memory
    ov::Tensor input_tensor(input_port.get_element_type(), input_port.get_shape(), blob.ptr(0));
    // Set input tensor for model with one input
    infer_request_Seg.set_input_tensor(input_tensor);

    // -------- Step 6. Start inference --------
    infer_request_Seg.infer();

    // -------- Step 7. Get the inference result --------
    auto output0 = infer_request_Seg.get_output_tensor(0); //output0
    auto output1 = infer_request_Seg.get_output_tensor(1); //otuput1
    auto output0_shape = output0.get_shape();
    auto output1_shape = output1.get_shape();
    std::cout << "The shape of output0:" << output0_shape << std::endl;
    std::cout << "The shape of output1:" << output1_shape << std::endl;

    // -------- Step 8. Postprocess the result --------
    Mat output_buffer(output1_shape[1], output1_shape[2], CV_32F, output1.data<float>());    // output_buffer 0:x 1:y  2 : w 3 : h   4--84 : class score  85--116 : mask pos
    Mat proto(32, 25600, CV_32F, output0.data<float>()); //[32,25600] 1 32 160 160
    transpose(output_buffer, output_buffer); //[8400,116]
    std::vector<int> class_ids;
    std::vector<float> class_scores;
    std::vector<Rect> boxes;
    std::vector<Mat> mask_confs;
    // Figure out the bbox, class_id and class_score
    for (int i = 0; i < output_buffer.rows; i++) {
        Mat classes_scores = output_buffer.row(i).colRange(4, 84);
        Point class_id;
        double maxClassScore;
        minMaxLoc(classes_scores, 0, &maxClassScore, 0, &class_id);

        if (maxClassScore > cof_threshold) {
            class_scores.push_back(maxClassScore);
            class_ids.push_back(class_id.x);
            float cx = output_buffer.at<float>(i, 0);
            float cy = output_buffer.at<float>(i, 1);
            float w = output_buffer.at<float>(i, 2);
            float h = output_buffer.at<float>(i, 3);

            int left = int((cx - 0.5 * w) * scale);
            int top = int((cy - 0.5 * h) * scale);
            int width = int(w * scale);
            int height = int(h * scale);

            cv::Mat mask_conf = output_buffer.row(i).colRange(84, 116);
            mask_confs.push_back(mask_conf);
            boxes.push_back(Rect(left, top, width, height));
        }
    }
    //NMS
    std::vector<int> indices;
    NMSBoxes(boxes, class_scores, cof_threshold, nms_area_threshold, indices);

    // -------- Visualize the detection results -----------
    cv::Mat rgb_mask = cv::Mat::zeros(src.size(), src.type());
    cv::Mat masked_img;
    cv::RNG rng;

    Mat dst_temp = src.clone();
    for (size_t i = 0; i < indices.size(); i++) 
    {
        // Visualize the objects
        int index = indices[i];
        int class_id = class_ids[index];
        rectangle(dst_temp, boxes[index], colors_seg[class_id % 16], 2, 8);
        std::string label = class_names[class_id] + ":" + std::to_string(class_scores[index]).substr(0, 4);
        Size textSize = cv::getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, 0);
        Rect textBox(boxes[index].tl().x, boxes[index].tl().y - 15, textSize.width, textSize.height + 5);
        cv::rectangle(dst_temp, textBox, colors_seg[class_id % 16], FILLED);
        putText(dst_temp, label, Point(boxes[index].tl().x, boxes[index].tl().y - 5), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255));

        // Visualize the Masks
        Mat m = mask_confs[i] * proto;
        for (int col = 0; col < m.cols; col++) {
            sigmoid_function(m.at<float>(0, col), m.at<float>(0, col));
        }
        cv::Mat m1 = m.reshape(1, 160); // 1x25600 -> 160x160
        int x1 = std::max(0, boxes[index].x);
        int y1 = std::max(0, boxes[index].y);
        int x2 = std::max(0, boxes[index].br().x);
        int y2 = std::max(0, boxes[index].br().y);
        int mx1 = int(x1 / scale * 0.25);
        int my1 = int(y1 / scale * 0.25);
        int mx2 = int(x2 / scale * 0.25);
        int my2 = int(y2 / scale * 0.25);

        cv::Mat mask_roi = m1(cv::Range(my1, my2), cv::Range(mx1, mx2));
        cv::Mat rm, det_mask;
        cv::resize(mask_roi, rm, cv::Size(x2 - x1, y2 - y1));

        for (int r = 0; r < rm.rows; r++) {
            for (int c = 0; c < rm.cols; c++) {
                float pv = rm.at<float>(r, c);
                if (pv > 0.5) {
                    rm.at<float>(r, c) = 1.0;
                }
                else {
                    rm.at<float>(r, c) = 0.0;
                }
            }
        }
        rm = rm * rng.uniform(0, 255);
        rm.convertTo(det_mask, CV_8UC1);
        if ((y1 + det_mask.rows) >= dst_temp.rows) {
            y2 = dst_temp.rows - 1;
        }
        if ((x1 + det_mask.cols) >= dst_temp.cols) {
            x2 = dst_temp.cols - 1;
        }

        cv::Mat mask = cv::Mat::zeros(cv::Size(dst_temp.cols, dst_temp.rows), CV_8UC1);
        det_mask(cv::Range(0, y2 - y1), cv::Range(0, x2 - x1)).copyTo(mask(cv::Range(y1, y2), cv::Range(x1, x2)));
        add(rgb_mask, cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255)), rgb_mask, mask);
        addWeighted(dst_temp, 0.5, rgb_mask, 0.5, 0, masked_img);
    }
    dst = masked_img.clone();

    return true;
}



// =====================姿态========================//
bool YoloModel::LoadPoseModel(const string& xmlName, string& device)
{
    //待优化，如何把初始化部分进行提取出来
   // -------- Step 1. Initialize OpenVINO Runtime Core --------
    ov::Core core;

    // -------- Step 2. Compile the Model --------
    compiled_model_Pose = core.compile_model(xmlName, device);

    // -------- Step 3. Create an Inference Request --------
    infer_request_Pose = compiled_model_Pose.create_infer_request();

    return true;
}




bool YoloModel::YoloPoseInfer(const Mat& src, double cof_threshold, double nms_area_threshold, Mat& dst, vector<Object>& vecObj)
{
    // -------- Step 4.Read a picture file and do the preprocess --------
    // Preprocess the image
    Mat letterbox_img;
    letterbox(src, letterbox_img);
    float scale = letterbox_img.size[0] / 640.0;
    Mat blob = blobFromImage(letterbox_img, 1.0 / 255.0, Size(640, 640), Scalar(), true);

    // -------- Step 5. Feed the blob into the input node of the Model -------
    // Get input port for model with one input
    auto input_port = compiled_model_Pose.input();
    // Create tensor from external memory
    ov::Tensor input_tensor(input_port.get_element_type(), input_port.get_shape(), blob.ptr(0));
    // Set input tensor for model with one input
    infer_request_Pose.set_input_tensor(input_tensor);

    // -------- Step 6. Start inference --------
    infer_request_Pose.infer();

    // -------- Step 7. Get the inference result --------
    auto output = infer_request_Pose.get_output_tensor(0);
    auto output_shape = output.get_shape();
    std::cout << "The shape of output tensor:" << output_shape << std::endl;

    // -------- Step 8. Postprocess the result --------
    float* data = output.data<float>();
    Mat output_buffer(output_shape[1], output_shape[2], CV_32F, data);
    transpose(output_buffer, output_buffer); //[8400,56]
    std::vector<int> class_ids;
    std::vector<float> class_scores;
    std::vector<Rect> boxes;
    std::vector<std::vector<float>> objects_keypoints;

    // //56: box[cx, cy, w, h] + Score + [17,3] keypoints
    for (int i = 0; i < output_buffer.rows; i++) {
        float class_score = output_buffer.at<float>(i, 4);

        if (class_score > cof_threshold) {
            class_scores.push_back(class_score);
            class_ids.push_back(0); //{0:"person"}
            float cx = output_buffer.at<float>(i, 0);
            float cy = output_buffer.at<float>(i, 1);
            float w = output_buffer.at<float>(i, 2);
            float h = output_buffer.at<float>(i, 3);
            // Get the box
            int left = int((cx - 0.5 * w) * scale);
            int top = int((cy - 0.5 * h) * scale);
            int width = int(w * scale);
            int height = int(h * scale);
            // Get the keypoints
            std::vector<float> keypoints;
            Mat kpts = output_buffer.row(i).colRange(5, 56);
            for (int i = 0; i < 17; i++) {
                float x = kpts.at<float>(0, i * 3 + 0) * scale;
                float y = kpts.at<float>(0, i * 3 + 1) * scale;
                float s = kpts.at<float>(0, i * 3 + 2);
                keypoints.push_back(x);
                keypoints.push_back(y);
                keypoints.push_back(s);
            }

            boxes.push_back(Rect(left, top, width, height));
            objects_keypoints.push_back(keypoints);
        }
    }
    //NMS
    std::vector<int> indices;
    NMSBoxes(boxes, class_scores, cof_threshold, nms_area_threshold, indices);

    dst = src.clone();
    // -------- Visualize the detection results -----------
    for (size_t i = 0; i < indices.size(); i++) {
        int index = indices[i];
        // Draw bounding box
        rectangle(dst, boxes[index], Scalar(0, 0, 255), 2, 8);
        std::string label = "Person:" + std::to_string(class_scores[index]).substr(0, 4);
        Size textSize = cv::getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, 0);
        Rect textBox(boxes[index].tl().x, boxes[index].tl().y - 15, textSize.width, textSize.height + 5);
        cv::rectangle(dst, textBox, Scalar(0, 0, 255), FILLED);
        putText(dst, label, Point(boxes[index].tl().x, boxes[index].tl().y - 5), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255));
        // Draw keypoints
        //std::vector<float> object_keypoints = objects_keypoints[index];
        //for (int i = 0; i < 17; i++)
        //{
        //    int x = std::clamp(int(object_keypoints[i * 3 + 0]), 0, dst.cols);
        //    int y = std::clamp(int(object_keypoints[i * 3 + 1]), 0, dst.rows);
        //    //Draw point
        //    circle(dst, Point(x, y), 5, posePalette[i], -1);
        //}
        // Draw keypoints-line

    }
    cv::Size shape = dst.size();
    plot_keypoints(dst, objects_keypoints, shape);
    return true;
}









void YoloModel::letterbox(const cv::Mat& source, cv::Mat& result)
{
    int col = source.cols;
    int row = source.rows;
    int _max = MAX(col, row);
    result = Mat::zeros(_max, _max, CV_8UC3);
    source.copyTo(result(Rect(0, 0, col, row)));
}



void YoloModel::sigmoid_function(float a, float& b) 
{
    b = 1. / (1. + exp(-a));
}



void YoloModel::plot_keypoints(cv::Mat& image, const std::vector<std::vector<float>>& keypoints, const cv::Size& shape)
{

    int radius = 5;
    bool drawLines = true;

    if (keypoints.empty()) {
        return;
    }

    std::vector<cv::Scalar> limbColorPalette;
    std::vector<cv::Scalar> kptColorPalette;

    for (int index : limbColorIndices) {
        limbColorPalette.push_back(posePalette[index]);
    }

    for (int index : kptColorIndices) {
        kptColorPalette.push_back(posePalette[index]);
    }

    for (const auto& keypoint : keypoints) {
        bool isPose = keypoint.size() == 51;  // numKeypoints == 17 && keypoints[0].size() == 3;
        drawLines &= isPose;

        // draw points
        for (int i = 0; i < 17; i++) {
            int idx = i * 3;
            int x_coord = static_cast<int>(keypoint[idx]);
            int y_coord = static_cast<int>(keypoint[idx + 1]);

            if (x_coord % shape.width != 0 && y_coord % shape.height != 0) {
                if (keypoint.size() == 3) {
                    float conf = keypoint[2];
                    if (conf < 0.5) {
                        continue;
                    }
                }
                cv::Scalar color_k = isPose ? kptColorPalette[i] : cv::Scalar(0, 0,
                    255);  // Default to red if not in pose mode
                cv::circle(image, cv::Point(x_coord, y_coord), radius, color_k, -1, cv::LINE_AA);
            }
        }
        // draw lines
        if (drawLines) {
            for (int i = 0; i < skeleton.size(); i++) {
                const std::vector<int>& sk = skeleton[i];
                int idx1 = sk[0] - 1;
                int idx2 = sk[1] - 1;

                int idx1_x_pos = idx1 * 3;
                int idx2_x_pos = idx2 * 3;

                int x1 = static_cast<int>(keypoint[idx1_x_pos]);
                int y1 = static_cast<int>(keypoint[idx1_x_pos + 1]);
                int x2 = static_cast<int>(keypoint[idx2_x_pos]);
                int y2 = static_cast<int>(keypoint[idx2_x_pos + 1]);

                float conf1 = keypoint[idx1_x_pos + 2];
                float conf2 = keypoint[idx2_x_pos + 2];

                // Check confidence thresholds
                if (conf1 < 0.5 || conf2 < 0.5) {
                    continue;
                }

                // Check if positions are within bounds
                if (x1 % shape.width == 0 || y1 % shape.height == 0 || x1 < 0 || y1 < 0 ||
                    x2 % shape.width == 0 || y2 % shape.height == 0 || x2 < 0 || y2 < 0) {
                    continue;
                }

                // Draw a line between keypoints
                cv::Scalar color_limb = limbColorPalette[i];
                cv::line(image, cv::Point(x1, y1), cv::Point(x2, y2), color_limb, 2, cv::LINE_AA);
            }
        }
    }
}
```

### 1.2 ov\_yolov8.h

```cpp
#pragma once
#ifdef OV_YOLOV8_EXPORTS
#define OV_YOLOV8_API _declspec(dllexport)
#else
#define OV_YOLOV8_API _declspec(dllimport)
#endif

#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <random>
#include <openvino/openvino.hpp> //openvino header file
#include <opencv2/opencv.hpp>    //opencv header file
using namespace cv;
using namespace std;
using namespace dnn;


// 定义输出结构体
typedef struct {
    float prob;
    cv::Rect rect;
    int classid;
}Object;


//定义类
class OV_YOLOV8_API YoloModel
{
public:
    YoloModel();
    ~YoloModel();
    //检测
    bool LoadDetectModel(const string& xmlName, string& device);
    bool YoloDetectInfer(const Mat& src, double cof_threshold, double nms_area_threshold, Mat& dst, vector<Object>& vecObj);

    //分类
    bool YoloClsInfer(const Mat& src, double cof_threshold, double nms_area_threshold, Mat& dst, vector<Object>& vecObj);
    bool LoadClsModel(const string& xmlName, string& device);

    //分割
    bool LoadSegModel(const string& xmlName, string& device);
    bool YoloSegInfer(const Mat& src, double cof_threshold, double nms_area_threshold, Mat& dst, vector<Object>& vecObj);

    //姿态
    bool LoadPoseModel(const string& xmlName, string& device);
    bool YoloPoseInfer(const Mat& src, double cof_threshold, double nms_area_threshold, Mat& dst, vector<Object>& vecObj);


private:
    ov::InferRequest infer_request_Detect;
    ov::CompiledModel compiled_model_Detect;

    ov::InferRequest infer_request_Cls;
    ov::CompiledModel compiled_model_Detect_Cls;

    ov::InferRequest infer_request_Seg;
    ov::CompiledModel compiled_model_Seg;

    ov::InferRequest infer_request_Pose;
    ov::CompiledModel compiled_model_Pose;

    //增加函数
    // Keep the ratio before resize
    void letterbox(const Mat& source, Mat& result);
    void sigmoid_function(float a, float& b);
    void plot_keypoints(cv::Mat& image, const std::vector<std::vector<float>>& keypoints, const cv::Size& shape);
};
```

## 2.Batch\_Test  
2.批量测试

### 2.1 Batch\_Test.cpp

```cpp
#include <iostream>
#include "ov_yolov8.h"
#pragma comment(lib,"..//x64//Release//OV_YOLOV8_DLL.lib")




int main(int argc, char* argv[])
{
    YoloModel yolomodel;
    string xmlName_Detect = "./yolov8/model/yolov8n.xml";
    string xmlName_Cls = "./yolov8/model/yolov8n-cls.xml";
    string xmlName_Seg = "./yolov8/model/yolov8n-seg.xml";
    string xmlName_Pose = "./yolov8/model/yolov8n-Pose.xml";
    string device = "GPU";
    bool initDetectflag = yolomodel.LoadDetectModel(xmlName_Detect, device);
    bool initClsflag = yolomodel.LoadClsModel(xmlName_Cls, device);
    bool initSegflag = yolomodel.LoadSegModel(xmlName_Seg, device);
    bool initPoseflag = yolomodel.LoadPoseModel(xmlName_Pose, device);
    if (initDetectflag == true)
    {
        cout << "检测模型初始化成功" << endl;
    }
    if (initClsflag == true)
    {
        cout << "分类模型初始化成功" << endl;
    }
    if (initSegflag == true)
    {
        cout << "分割模型初始化成功" << endl;
    }
    if (initPoseflag == true)
    {
        cout << "姿态模型初始化成功" << endl;
    }
    // 读取图像
    Mat img_Detect = cv::imread("./yolov8/img/bus.jpg");
    Mat img_Cls = img_Detect.clone();
    Mat img_Seg = img_Detect.clone();
    Mat img_Pose = img_Detect.clone();

    // 检测推理
    Mat dst_detect;
    double cof_threshold_detect  = 0.25;
    double nms_area_threshold_detect = 0.5;
    vector<Object> vecObj = {};
    bool InferDetectflag = yolomodel.YoloDetectInfer(img_Detect, cof_threshold_detect, nms_area_threshold_detect, dst_detect, vecObj);


    // 分类推理
    Mat dst_cls;
    double cof_threshold_Cls = 0.25;
    double nms_area_threshold_Cls = 0.5;
    vector<Object> vecObj_cls = {};
    bool InferClsflag = yolomodel.YoloClsInfer(img_Cls, cof_threshold_Cls, nms_area_threshold_Cls, dst_cls, vecObj_cls);


    // 分割推理
    Mat dst_seg;
    double cof_threshold_Seg = 0.25;
    double nms_area_threshold_Seg = 0.5;
    vector<Object> vecObj_seg = {};
    bool InferSegflag = yolomodel.YoloSegInfer(img_Seg, cof_threshold_Seg, nms_area_threshold_Seg, dst_seg, vecObj_seg);

    // 姿态推理
    Mat dst_pose;
    double cof_threshold_Pose = 0.25;
    double nms_area_threshold_Pose = 0.5;
    vector<Object> vecObj_Pose = {};
    bool InferPoseflag = yolomodel.YoloPoseInfer(img_Pose, cof_threshold_Pose, nms_area_threshold_Pose, dst_pose, vecObj_Pose);

    namedWindow("dst_pose", WINDOW_NORMAL);
    //imshow("dst_detect", dst_detect);
    imshow("dst_pose", dst_pose);
    waitKey(0);
    destroyAllWindows();
    return 0;
}
```

## 3\. 完整工程

[https://download.csdn.net/download/qq\_44747572/88580524](https://link.zhihu.com/?target=https%3A//download.csdn.net/download/qq_44747572/88580524)

  

本文来自本人CSDN博客：[【模型c++部署】yolov8（检测、分类、分割、姿态）使用openvino进行部署-CSDN博客](https://link.zhihu.com/?target=https%3A//blog.csdn.net/qq_44747572/article/details/134309299%3Fspm%3D1001.2014.3001.5502)  
本文来自本人CSDN博客[](https://link.zhihu.com/?target=https%3A//blog.csdn.net/qq_44747572/article/details/134309299%3Fspm%3D1001.2014.3001.5502)

发布于 2023-12-06 08:11・IP 属地中国香港