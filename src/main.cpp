#define BOOST_BIND_NO_PLACEHOLDERS
#include <chrono>
#include <functional>
#include <memory>
#include <string>

#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include "pointmatcher/PointMatcher.h"
#include <pcl/features/normal_3d_omp.h>
#include "pointmatcher/IO.h"
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include <pcl/octree/octree_search.h>
//#include <pcl_ros/point_cloud.hpp>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl_ros/transforms.hpp>
#include <pcl/io/pcd_io.h>
//#include <pcl/features/normal_3d_omp.h>
#include <pcl/common/transforms.h>
//#include <pcl/filters/voxel_grid.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types.h>

using namespace std::chrono_literals;
using std::placeholders::_1;

/* This example creates a subclass of Node and uses std::bind() to register a
* member function as a callback from the timer. */

class LidarMapper : public rclcpp::Node
{
protected:

    /* ------------------------------ variables------------------------------------------------------------------------- */

    bool init;
    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr publisher_;

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subscriber_;

    size_t count_;
    std::shared_ptr<tf2_ros::TransformListener> transform_listener_{nullptr};
    std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
    Eigen::Matrix4d base_tf;
    pcl::PointCloud<pcl::PointXYZI>::Ptr rolling_cloud, rolling_cloud2;
    Eigen::Vector3d prev_position;
    Eigen::Matrix4d map_tf;

    float leaf_size;
    bool computeProbDynamic;
    float priorDynamic;
    std::string icpParamPath;
    std::string icpInputParamPath;
    std::string icpPostParamPath;

    Eigen::Matrix4d lastCorrection;

    typedef PointMatcher<float> PM;
    //typedef PointMatcherIO<float> PMIO;
    typedef PM::TransformationParameters TP;
    typedef PM::DataPoints DP;
    DP mapPointCloud;

    std::shared_ptr<PM::Transformation> rigidTrans;

    /* ------------------------------functions -------------------------------------------------------------------------*/


    void icpFn(DP& newCloud, DP& mapPointCloud, Eigen::Matrix4d& correction, Eigen::Matrix4d& prior, std::string icpParamPath, std::string icpInputParamPath, std::string icpPostParamPath)
    {

        correction=Eigen::Matrix4d::Identity();

        // Main algorithm definition
        PM::ICP icp;
        PM::DataPointsFilters inputFilters;
        PM::DataPointsFilters mapPostFilters;



        if(!icpInputParamPath.empty())
        {
            std::ifstream ifs(icpInputParamPath.c_str());
            inputFilters = PM::DataPointsFilters(ifs);
            std::cout<<"loaded input filter yaml!"<<std::endl;
            ifs.close();
        }

        if(!icpPostParamPath.empty())
        {
            std::ifstream ifs(icpPostParamPath.c_str());
            mapPostFilters = PM::DataPointsFilters(ifs);
            std::cout<<"loaded post filter yaml!"<<std::endl;
            ifs.close();
        }

        inputFilters.apply(newCloud);

        std::cout<<"map pc has "<<mapPointCloud.getNbPoints()<<"points"<<std::endl;
        mapPostFilters.apply(mapPointCloud);

        if(!icpParamPath.empty())
        {
            std::ifstream ifs(icpParamPath.c_str());
            icp.loadFromYaml(ifs);
            std::cout<<"loaded icp yaml!"<<std::endl;
            ifs.close();
        }
        else
        {
            icp.setDefault();
        }






        if (newCloud.getNbPoints()>50){


            try
            {
                // We use the last transformation as a prior
                // this assumes that the point clouds were recorded in
                // sequence.
                const TP prior0 = prior.cast<float>();


                TP T_to_map_from_new = icp(newCloud, mapPointCloud, prior0);

                T_to_map_from_new = rigidTrans->correctParameters(T_to_map_from_new);

                correction=T_to_map_from_new.cast <double> ();

            }
            catch (PM::ConvergenceError& error)
            {
                std::cout << "ERROR PM::ICP failed to converge: " << std::endl;
                std::cout << "   " << error.what() << std::endl;

            }

        }
    }

    void transform_cloud(pcl::PointCloud<pcl::PointXYZ>& cloudIn,Eigen::Matrix4d pose){

        for (unsigned int j=0; j<cloudIn.points.size(); j++){

            double xx=cloudIn.points[j].x; double yy=cloudIn.points[j].y; double zz=cloudIn.points[j].z;
            // double wx, wy, wz;
            //  double intensity=cloudIn.points[j].intensity;
            Eigen::Vector4d pcPoints(xx,yy,zz,1.0);
            Eigen::Vector4d pcPointsTransformed=pose*pcPoints;
            pcl::PointXYZ p;
            p.x=pcPointsTransformed[0];
            p.y=pcPointsTransformed[1];
            p.z=pcPointsTransformed[2];

            // p.r=cloudIn.points[j].r;
            // p.g=cloudIn.points[j].g;
            //p.b=cloudIn.points[j].b;



            // p.intensity=cloudIn.points[j].intensity;

            cloudIn.points[j]=p;
        }

    }

    void denoise_map(pcl::PointCloud<pcl::PointXYZ>::Ptr& temp_cloud){

        pcl::StatisticalOutlierRemoval<pcl::PointXYZ> statFilter;
        statFilter.setInputCloud (temp_cloud);
        statFilter.setMeanK (20);
        statFilter.setStddevMulThresh (5);
        statFilter.filter (*temp_cloud);

    }

    void octree_voxelize(pcl::PointCloud<pcl::PointXYZ>::Ptr& temp_cloud, float leaf_size=0.015){

        pcl::octree::OctreePointCloud<pcl::PointXYZ> octree_cloud (leaf_size);
        octree_cloud.setInputCloud (temp_cloud);
        octree_cloud.addPointsFromInputCloud ();
        clear_cloud(temp_cloud);
        octree_cloud.getOccupiedVoxelCenters(temp_cloud->points);

    }
    void clear_cloud(pcl::PointCloud<pcl::PointXYZ>::Ptr& temp_cloud){

        temp_cloud->clear();
        temp_cloud.reset(new pcl::PointCloud<pcl::PointXYZ>);
    }


    DP icp_process(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud)
    {


        using namespace PointMatcherSupport;
        pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> ne;
        ne.setInputCloud (cloud);
        pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ> ());
        ne.setSearchMethod (tree);
        pcl::PointCloud<pcl::Normal>::Ptr cloud_normals (new pcl::PointCloud<pcl::Normal>);
        ne.setRadiusSearch (0.005);
        ne.compute (*cloud_normals);

        DP data;

        Eigen::MatrixXf dataNormals(3,cloud_normals->getMatrixXfMap(3,4,0).row(0).size());
        dataNormals.row(0)=cloud_normals->getMatrixXfMap(3,4,0).row(0);
        dataNormals.row(1)=cloud_normals->getMatrixXfMap(3,4,0).row(1);
        dataNormals.row(2)=cloud_normals->getMatrixXfMap(3,4,0).row(2);
        Eigen::MatrixXf datax(1,cloud->getMatrixXfMap(3,4,0).row(0).size());
        Eigen::MatrixXf datay(1,cloud->getMatrixXfMap(3,4,0).row(1).size());
        Eigen::MatrixXf dataz(1,cloud->getMatrixXfMap(3,4,0).row(2).size());
        datax=cloud->getMatrixXfMap(3,4,0).row(0);
        datay=cloud->getMatrixXfMap(3,4,0).row(1);
        dataz=cloud->getMatrixXfMap(3,4,0).row(2);

        data.addFeature("x", datax);
        data.addFeature("y", datay);
        data.addFeature("z", dataz);
        data.addDescriptor("normals", dataNormals);

        if(computeProbDynamic)

        {
            data.addDescriptor("probabilityDynamic", PM::Matrix::Constant(1, data.features.cols(), priorDynamic));

        }
        return data;

    }



    pcl::PointCloud<pcl::PointXYZ> boundingBox(pcl::PointCloud<pcl::PointXYZ>::Ptr temp_cloud, double insideTh, double outsideTh){

        pcl::PointCloud<pcl::PointXYZ> p;

        for (unsigned int j=0; j<temp_cloud->points.size(); j++){

            double xx=temp_cloud->points[j].x;
            double yy=temp_cloud->points[j].y;
            double zz=temp_cloud->points[j].z;

            if (  (sqrt (xx*xx+yy*yy+zz*zz)<insideTh ) || sqrt (xx*xx+yy*yy+zz*zz)>outsideTh ) {  //hemispherical box
                //do nothing for now
            }
            else
            {
                p.points.push_back(temp_cloud->points[j]);
            }
        }
        return p;
    }




    void lidar_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
    {


        std::string filename;
        //filename.precision(16);
        double filename_d= (double(msg->header.stamp.sec)+double(msg->header.stamp.nanosec)*1e-9);

        filename=std::to_string(filename_d);

        std::cout <<"filename is "<<filename<<" "<<filename<<std::endl;

        std::string currentPath="/home/georges/export";  // TODO: make it a rosparam
        pcl::PCLPointCloud2::Ptr pc2 (new pcl::PCLPointCloud2 ());
        pcl_conversions::toPCL(*msg,*pc2);
        std::vector<int> indices;
        pcl::PointCloud<pcl::PointXYZ>::Ptr temp_cloud (new pcl::PointCloud<pcl::PointXYZ>);
        pcl::fromPCLPointCloud2(*pc2,*temp_cloud);

        //pcl::removeNaNFromPointCloud(*temp_cloud, *temp_cloud, indices);
        octree_voxelize(temp_cloud, leaf_size);


        DP data=icp_process(temp_cloud); //calculate normals and prepare data structures

        if(!init){
            mapPointCloud=data;
        }

        std::cout<<"map pc has "<<mapPointCloud.getNbPoints()<<" points"<<std::endl;

        std::cout<<"scan pc has "<<data.getNbPoints()<<" points"<<std::endl;

        Eigen::Matrix4d prior=lastCorrection;

        Eigen::Matrix4d correction;

        icpFn(data, mapPointCloud, correction, prior, icpParamPath, icpInputParamPath, icpPostParamPath);

        if (!correction.isIdentity()){
            lastCorrection=correction;
        }

        //else
        //{
        //    lastCorrection=Eigen::Matrix4d::Identity();
        //}

        std::cout<<"prior is \n"<<prior<<std::endl;
        std::cout<<"correction is \n"<<lastCorrection<<std::endl;

        if (init && !correction.isIdentity()){

            data = rigidTrans->compute(  data, correction.cast<float>() );

            mapPointCloud.concatenate(data);

        }

        else
        {
            if (init){
                std::cout<<"ICP failed"<<std::endl;
            }
            else{

                data = rigidTrans->compute(  data, correction.cast<float>() );

                mapPointCloud.concatenate(data);

            }
        }


        //sensor_msgs::PointCloud2 pointMatcherCloudToRosMsg<double>(mapPointCloud, const std::stri, const ros::Time& stamp);
        Eigen::MatrixXf xx=mapPointCloud.getFeatureViewByName("x");
        Eigen::MatrixXf yy=mapPointCloud.getFeatureViewByName("y");
        Eigen::MatrixXf zz=mapPointCloud.getFeatureViewByName("z");

        pcl::PointCloud<pcl::PointXYZ>::Ptr pub_cloud (new pcl::PointCloud<pcl::PointXYZ>);

        for (int i=0; i<xx.cols();i++){

            pcl::PointXYZ p;

            p.x=xx(i);
            p.y=yy(i);
            p.z=zz(i);

            pub_cloud->points.push_back(p);
        }

        //std::cout<<xx<<std::endl;

        std::cout<<"Saving "<<pub_cloud->points.size()<<" points"<<std::endl;

        pub_cloud->height=1;
        pub_cloud->width=pub_cloud->points.size();

        sensor_msgs::msg::PointCloud2 pc2_msg_;

        pcl::toROSMsg(*pub_cloud, pc2_msg_);

        pc2_msg_.header.frame_id = "laser_sensor_frame";

        pc2_msg_.header.stamp = msg->header.stamp;

        publisher_->publish(pc2_msg_);

        // std::string fullPath= currentPath + "/" + filename + ".pcd";

        //  pcl::io::savePCDFileASCII (fullPath, *temp_cloud);

        init=true;
        // rclcpp::sleep_for(std::chrono::nanoseconds(100ms));
    }

public:
    LidarMapper()
        : Node("ros2_live_icp"), count_(0)
    {

        subscriber_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
                    "points", rclcpp::SensorDataQoS(), std::bind(&LidarMapper::lidar_callback, this, _1));

        publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("map_out", 10);

        rigidTrans = PM::get().REG(Transformation).create("RigidTransformation");

        lastCorrection=Eigen::Matrix4d::Identity();

        init=false;
        computeProbDynamic=true;
        leaf_size=0.025;

        icpParamPath="/home/georges/catkin_ws/src/ros2_live_icp/icpConfig/icp_param.yaml";
        icpInputParamPath="/home/georges/catkin_ws/src/ros2_live_icp/icpConfig/input_filters.yaml";
        icpPostParamPath="/home/georges/catkin_ws/src/ros2_live_icp/icpConfig/mapPost_filters.yaml";

        rclcpp::TimerBase::SharedPtr timer_{nullptr};
        tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
        transform_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

    }

};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<LidarMapper>());
    rclcpp::shutdown();
    return 0;
}
