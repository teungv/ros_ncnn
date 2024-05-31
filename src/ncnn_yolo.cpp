#include "ros_ncnn/ncnn_utils.h"
#include "ros_ncnn/ncnn_yolo.h"

int ncnnYolo::detect(const cv::Mat& bgr, std::vector<Object>& objects, uint8_t n_threads)
{
  const int target_size = 352;

  int img_w = bgr.cols;
  int img_h = bgr.rows;

  ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR, bgr.cols, bgr.rows, target_size, target_size);

  const float mean_vals[3] = {0.f, 0.f, 0.f};
  const float norm_vals[3] = {1/255.f, 1/255.f, 1/255.f};
  in.substract_mean_normalize(mean_vals, norm_vals);

  ncnn::Extractor ex = neuralnet.create_extractor();
  ex.set_num_threads(n_threads);

  ex.input("data", in);

  //     // List all layer names
  // for (int i = 0; i < neuralnet.layers().size(); i++)
  // {
  //     const ncnn::Layer* layer = neuralnet.layers()[i];
  //     printf("Layer %d: %s\n", i, layer->name.c_str());
  // }

  ncnn::Mat out;
//   std::cout << "I AM HERE"; 


  ex.extract("output", out);

  objects.clear();
  for (int i=0; i<out.h; i++)
  {
      const float* values = out.row(i);

      Object object;
      object.label = values[0];
      object.prob = values[1];
      object.rect.x = values[2] * img_w;
      object.rect.y = values[3] * img_h;
      object.rect.width = values[4] * img_w - object.rect.x;
      object.rect.height = values[5] * img_h - object.rect.y;

      objects.push_back(object);
  }

  return 0;
}

void ncnnYolo::draw(const cv::Mat& bgr, const std::vector<Object>& objects, double dT)
{
    cv::Mat image = bgr.clone();
    std::cout << objects.size();

    for (size_t i = 0; i < objects.size(); i++)
    {
        const Object& obj = objects[i];

        cv::rectangle(image, obj.rect, cv::Scalar(255, 0, 0));

        char text[256];
        sprintf(text, "%s %.1f%%", class_names[obj.label], obj.prob * 100);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        int x = obj.rect.x;
        int y = obj.rect.y - label_size.height - baseLine;
        if (y < 0)
            y = 0;
        if (x + label_size.width > image.cols)
            x = image.cols - label_size.width;

        cv::rectangle(image, cv::Rect(cv::Point(x, y),
                                      cv::Size(label_size.width, label_size.height + baseLine)),
                      cv::Scalar(255, 255, 255), -1);

        cv::putText(image, text, cv::Point(x, y + label_size.height),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
    }

    cv::putText(image, std::to_string(1/dT)+" Hz", cv::Point(20, 20), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255));
    cv::imshow("YOLO", image);
    cv::waitKey(1);
}
