#include <ros/package.h>
#include <ros/ros.h>
#include <ros/console.h>
#include <image_transport/image_transport.h>
#include <sensor_msgs/image_encodings.h>
#include <cv_bridge/cv_bridge.h>

#include "ros_ncnn/ncnn_config.h"
#ifdef GPU_SUPPORT
  #include "gpu.h"
  #include "ros_ncnn/gpu_support.h"
#endif

/////////////////////////////////
#include "ros_ncnn/ncnn_yolo.h"
#include "ros_ncnn/Object.h"
#include "ros_ncnn/ObjectArray.h"
ncnnYolo engine;
/////////////////////////////////

ros_ncnn::Object objMsg;
ros::Publisher obj_pub;
image_transport::Publisher image_pub;
std::vector<Object> objects;
cv_bridge::CvImagePtr cv_ptr;
ros::Time last_time;
bool display_output;
double prob_threshold;
bool enable_gpu;

void imageCallback(const sensor_msgs::ImageConstPtr& msg, int n_threads)
{ 

  cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
  cv::Mat cv_img;
  cv_img = cv_ptr->image;
  image_pub.publish(cv_ptr->toImageMsg());
  ros_ncnn::ObjectArray bboxarray;


  try {
    ros::Time current_time = ros::Time::now();  
    engine.detect(cv_ptr->image, objects, n_threads); // create the objects from the model
    
    for (size_t i = 0; i < objects.size(); i++)
    {
        const Object& obj = objects[i];
        if (obj.prob > prob_threshold)
        {
          ROS_INFO("%d = %.5f at %.2f %.2f %.2f x %.2f", obj.label, obj.prob,
          obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height);
          objMsg.header.seq++;
          objMsg.header.stamp = current_time;
          objMsg.probability = obj.prob;
          objMsg.label = class_names[obj.label];
          objMsg.boundingbox.position.x = obj.rect.x;
          objMsg.boundingbox.position.y = obj.rect.y;
          objMsg.boundingbox.size.x = obj.rect.width;
          objMsg.boundingbox.size.y = obj.rect.height;

          bboxarray.objectarray.push_back(objMsg);


          // cv::Rect rect(obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height);
          // cv::rectangle(cv_ptr->image, rect, cv::Scalar(0, 255, 0), 2);
          // cv::putText(cv_ptr->image, objMsg.label, cv::Point(obj.rect.x, obj.rect.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
        }
    }

    obj_pub.publish(bboxarray);

    if (display_output) {
      
      engine.draw(cv_ptr->image, objects, (current_time-last_time).toSec());
    }
    last_time = current_time;
  }
  catch (cv_bridge::Exception& e) {
    ROS_ERROR("CV bridge exception: %s", e.what());
    return;
  }
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "yolo_node"); /**/
  ros::NodeHandle nhLocal("~");
  ros::NodeHandle n;
  std::string node_name = ros::this_node::getName();
  int gpu_device;
  nhLocal.param("gpu_device", gpu_device, 0);
  nhLocal.param("enable_gpu", enable_gpu, true);


#ifndef GPU_SUPPORT
  ROS_WARN_STREAM(node_name << " running on CPU");
#endif
#ifdef GPU_SUPPORT
  ROS_INFO_STREAM(node_name << " with GPU_SUPPORT, selected gpu_device: " << gpu_device);
  g_vkdev = ncnn::get_gpu_device(selectGPU(gpu_device));
  g_blob_vkallocator = new ncnn::VkBlobAllocator(g_vkdev);
  g_staging_vkallocator = new ncnn::VkStagingAllocator(g_vkdev);
  engine.neuralnet.opt.use_vulkan_compute = enable_gpu;
  engine.neuralnet.set_vulkan_device(g_vkdev);
  
#endif
  
  
  const std::string package_name = "ros_ncnn";
  std::string path = ros::package::getPath(package_name)+("/assets/models/");
  ROS_INFO("Assets path: %s", path.c_str());

  std::string model_file, param_file;

  nhLocal.param("model_file", model_file, std::string("mobilenetv2_yolov3.bin"));
  nhLocal.param("param_file", param_file, std::string("mobilenetv2_yolov3.param"));
  engine.neuralnet.load_param((path+param_file).c_str());
  engine.neuralnet.load_model((path+model_file).c_str());
  ROS_INFO("Loaded: %s", model_file.c_str());
  
  int num_threads;
  nhLocal.param("num_threads", num_threads, ncnn::get_cpu_count());
  engine.neuralnet.opt.num_threads=num_threads;
  nhLocal.param("display_output", display_output, true);

  obj_pub = n.advertise<ros_ncnn::ObjectArray>(node_name+"/objects", 10);
  image_transport::ImageTransport it(n);
  image_transport::Subscriber video = it.subscribe("/camera/image_raw", 1, boost::bind(&imageCallback, _1, num_threads));
  image_pub = it.advertise(node_name+"/annotated_image", 10);

#ifdef GPU_SUPPORT
  
#endif
  while (ros::ok()) {
    ros::spinOnce();
  }
#ifdef GPU_SUPPORT
 
#endif

  return 0;
}
