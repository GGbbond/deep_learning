#include <openvino/openvino.hpp>
#include <opencv2/opencv.hpp>

#include<string>
#include<iostream>

float cof_threshold = 0.3;
float nms_area_threshold = 0.5;
std::string VIDOE_PATH = "/home/fuziming/MA/Deep_learing/用openvino部署模型/雷达test/5/Screenrecorder-2024-02-27-21-01-00-24.mp4";
// std::string VIDOE_PATH = "/home/fuziming/MA/Deep_learing/用openvino部署模型/雷达test/5/平移旋转 (1).mp4";
std::string MODEL_PATH = "/home/fuziming/MA/Deep_learing/用openvino部署模型/雷达test/5/best_openvino_model/best.xml";
std::string DEVICE = "GPU";

cv::Mat img1;
cv::VideoCapture capture;

void letterbox(const cv::Mat& source, cv::Mat& result)
{
    int col = source.cols;
    int row = source.rows;
    int _max = MAX(col, row);
    result = cv::Mat::zeros(_max, _max, CV_8UC3);
    source.copyTo(result(cv::Rect(0, 0, col, row)));
}


int main() {

	capture.open(VIDOE_PATH);
	if (!capture.isOpened()) {
		printf("could not load video data...\n");
		return -1;
	}

	int frames = capture.get(cv::CAP_PROP_FRAME_COUNT);//获取视频针数目(一帧就是一张图片)

    //1.Create Runtime Core
	ov::Core core;

	//2.Compile the model
	ov::CompiledModel compiled_model = core.compile_model(MODEL_PATH, DEVICE);

	//3.Create inference request
	ov::InferRequest infer_request = compiled_model.create_infer_request();

    float gamma = 0.5;

    // 将图像像素值应用伽马转换来降低亮度
    cv::Mat darkenedImage;


    for(int i = 0; i < frames; i++)
    {
        capture >> img1;

        cv::Mat letterbox_img;
        letterbox(img1,letterbox_img);
        float scale = letterbox_img.size[0] / 640.0;
		cv::Mat blob = cv::dnn::blobFromImage(letterbox_img, 1.0 / 255.0, cv::Size(640, 640), cv::Scalar(), true);
        
        auto input_port = compiled_model.input();

		ov::Tensor input_tensor(input_port.get_element_type(), input_port.get_shape(), blob.ptr(0));

		infer_request.set_input_tensor(input_tensor);

        infer_request.infer();

		//Get output
		auto output = infer_request.get_output_tensor(0);
		auto output_shape = output.get_shape();
        // std::cout<<output_shape<<std::endl;

        float* data = output.data<float>();
        
		cv::Mat output_buffer(output_shape[1], output_shape[2], CV_32F, data);
		transpose(output_buffer, output_buffer); //[8400,23]
        cv::Mat dst = img1.clone();
        std::vector<int> class_ids;
        int class_id;

        // std::cout<<output_buffer<<std::endl;

        std::vector<float> class_scores;
        std::vector<cv::Rect> boxes;

        for (int i = 0; i < output_buffer.rows; i++) {
            float class_score = output_buffer.at<float>(i, 4);
            if (class_score > cof_threshold) {
                class_scores.push_back(class_score);
                class_ids.push_back(i); 
                float cx = output_buffer.at<float>(i, 0);
                float cy = output_buffer.at<float>(i, 1);
                float w = output_buffer.at<float>(i, 2);
                float h = output_buffer.at<float>(i, 3);
                // Get the box
                int left = int((cx - 0.5 * w) * scale);
                int top = int((cy - 0.5 * h) * scale);
                int width = int(w * scale);
                int height = int(h * scale);

                boxes.push_back(cv::Rect(left, top, width, height));

            }

        }

        //NMS
        std::vector<int> indices;
        if(boxes.size()>0){
            cv::dnn::NMSBoxes(boxes, class_scores, cof_threshold, nms_area_threshold, indices);
        }


        // -------- Visualize the detection results -----------
        for (size_t i = 0; i < indices.size(); i++) {
            int index = indices[i];

             // Draw bounding box
                cv::rectangle(dst, boxes[index], cv::Scalar(0, 0, 255), 2, 8);
                std::string label = std::to_string(class_id) + ":" + std::to_string(class_scores[index]).substr(0, 4);
                cv::Size textSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, 0);
                cv::Rect textBox(boxes[index].tl().x, boxes[index].tl().y - 15, textSize.width, textSize.height + 5);
                cv::rectangle(dst, textBox, cv::Scalar(0, 0, 255), cv::FILLED);
                cv::putText(dst, label, cv::Point(boxes[index].tl().x, boxes[index].tl().y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255));

        }
        if (img1.empty())break;
		cv::imshow("dst",dst);
		// cv::waitKey(0);
		if (cv::waitKey(1) >= 0) break;
    }

    capture.release();

	return 0;
}