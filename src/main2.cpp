#include <ros/ros.h>
#include <tf/transform_broadcaster.h>
#include <sensor_msgs/Imu.h>
#include <tf/LinearMath/Matrix3x3.h>
#include <geometry_msgs/Vector3.h>
#include <geometry_msgs/PoseStamped.h>
#include <nav_msgs/OccupancyGrid.h>
#include <nav_msgs/Path.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include "sensor_msgs/PointCloud2.h"
//#include <crawler_msgs/LiveTexture.h>
#include <crawler_msgs/TextureTriggerSrv.h>
#include <crawler_msgs/PoseTriggerSrv.h>
#include "std_msgs/Float64.h"
#include "nav_msgs/Odometry.h"
#include <pcl_ros/point_cloud.h>
#include <pcl_ros/transforms.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/point_types.h>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <image_transport/image_transport.h>
#include "vector"
#include "pointmatcher/PointMatcher.h"
#include "pointmatcher/IO.h"
#include <cv_bridge/cv_bridge.h>
#include <opencv2/highgui/highgui.hpp>
#include "DMeansT.h"
#include <stdio.h>
#include <math.h>
#include <fstream>
#include <tf/transform_broadcaster.h>
using namespace std;
using namespace Clustering;

#define PI 3.14159265
typedef std::pair<double,double> DI;
typedef std::vector<DI> DataSet;
typedef DataSet::const_iterator DataSetIterator;



template <class stream>
stream & operator<<(stream & out, const DI & d) {
    out << d.first << " " << d.second;
    return out;
}


class Mapper

{
protected:
    typedef PointMatcher<float> PM;
    //typedef PointMatcherIO<float> PMIO;
    typedef PM::TransformationParameters TP;
    typedef PM::DataPoints DP;
    DP mapPointCloud;
    float priorDynamic;
    double lastTime;
    int iteration;
    tf::Quaternion q0;
    tf::Vector3 t0;
    Eigen::Quaternion<double> rotation;
    Eigen::Matrix3d rotM;
    Eigen::Matrix4d ifmTransform, zeroTransform, lastCorrection;
    //  nav_msgs::OccupancyGrid globalOutlierMap;  //obstacles occupancy grid
    float leaf_size;
    bool doIcp, skipRansac;
    double textureResolution, rotationTuning;
    double gx, gy, gz, gxx, gyy, gzz, aa, bb, cc, dd, x, y, n1, n2, n3;
    double x0Init=0;
    double x0Prev=0;
    cv::Mat texture;
    image_transport::Subscriber textureSub;
    double startStamp, endStamp;
    bool lockMap;
    std_msgs::Header lastHeader;
    double xOffset, yOffset;
    nav_msgs::Path path;
    image_transport::Publisher obsImagePub;
    double y0Init=0;
    double y0Prev=0;
    double mapBuildTime, outsideTh;

    Eigen::Matrix4d capturedPose, lastEkfPose, lastCapturedPose, prevPose, prevMapPose;

    double yawInit=0;

    int erosion_elem = 0;
    int dilation_elem = 0;

    int erosion_size = 0;

    int dilation_size =2;

    std::string icpParamPath;
    std::string icpInputParamPath;
    std::string icpPostParamPath;
    bool init, useTf, textureTrigger;
    bool computeProbDynamic;
    ros::NodeHandle n;
    // Rigid transformation
    std::shared_ptr<PM::Transformation> rigidTrans;


    ros::Publisher localOutlierPub, ransacPlanePub, localInlierPub, scanFilteredPub, posePub, odomPub, globalMapPub, pathPub, localMapPub;
    image_transport::ImageTransport it;
    message_filters::Subscriber<sensor_msgs::Imu> imu_sub;
    message_filters::Subscriber<nav_msgs::Odometry> odom_sub;
    message_filters::Subscriber<sensor_msgs::PointCloud2> pc_sub;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr globalMap, globalOutlierMap, localMap;
    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Imu, sensor_msgs::PointCloud2> MySyncPolicy;
    typedef message_filters::Synchronizer<MySyncPolicy> Sync;
    boost::shared_ptr<Sync> sync;
    tf::StampedTransform ifmTf, zeroTf;
    tf::TransformListener listener1, listener2;
    tf::TransformBroadcaster broadcaster;

    pcl::PointCloud<pcl::PointXYZRGB> boundingBox(pcl::PointCloud<pcl::PointXYZRGB>::Ptr temp_cloud, double insideTh, double outsideTh){

        pcl::PointCloud<pcl::PointXYZRGB> p;

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


    void icpFn(DP& newCloud, DP& mapPointCloud, Eigen::Matrix4d& correction, Eigen::Matrix4d& prior, std::string icpParamPath, std::string icpInputParamPath, std::string icpPostParamPath)
    {

        correction=Eigen::Matrix4d::Identity();
        // Rigid transformation
        std::shared_ptr<PM::Transformation> rigidTrans;
        rigidTrans = PM::get().REG(Transformation).create("RigidTransformation");

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
                // const TP prior = T_to_map_from_new*initialEstimate.matrix().cast<float>();
                //const TP prior= TP::Identity(4,4);

                TP T_to_map_from_new = icp(newCloud, mapPointCloud, prior0);

                T_to_map_from_new = rigidTrans->correctParameters(T_to_map_from_new);
                newCloud = rigidTrans->compute(newCloud, T_to_map_from_new);
                // mapPointCloud.concatenate(newCloud);

                correction=T_to_map_from_new.cast <double> ();

            }
            catch (PM::ConvergenceError& error)
            {
                std::cout << "ERROR PM::ICP failed to converge: " << std::endl;
                std::cout << "   " << error.what() << std::endl;

            }

        }
    }

    void transform_cloud(pcl::PointCloud<pcl::PointXYZRGB>& cloudIn,Eigen::Matrix4d pose){

        for (unsigned int j=0; j<cloudIn.points.size(); j++){

            double xx=cloudIn.points[j].x; double yy=cloudIn.points[j].y; double zz=cloudIn.points[j].z;
            // double wx, wy, wz;
            //  double intensity=cloudIn.points[j].intensity;
            Eigen::Vector4d pcPoints(xx,yy,zz,1.0);
            Eigen::Vector4d pcPointsTransformed=pose*pcPoints;
            pcl::PointXYZRGB p;
            p.x=pcPointsTransformed[0];
            p.y=pcPointsTransformed[1];
            p.z=pcPointsTransformed[2];

            p.r=cloudIn.points[j].r;
            p.g=cloudIn.points[j].g;
            p.b=cloudIn.points[j].b;



            // p.intensity=cloudIn.points[j].intensity;

            cloudIn.points[j]=p;
        }

    }


    void denoise_map(pcl::PointCloud<pcl::PointXYZRGB>::Ptr& temp_cloud){

        pcl::StatisticalOutlierRemoval<pcl::PointXYZRGB> statFilter;
        statFilter.setInputCloud (temp_cloud);
        statFilter.setMeanK (20);
        statFilter.setStddevMulThresh (5);
        statFilter.filter (*temp_cloud);

    }
    void voxelize(pcl::PointCloud<pcl::PointXYZRGB>::Ptr& temp_cloud, float leaf_size=0.01){

        pcl::VoxelGrid<pcl::PointXYZRGB> sor;
        sor.setInputCloud (temp_cloud);
        sor.setMinimumPointsNumberPerVoxel(2);
        sor.setLeafSize (leaf_size,leaf_size,leaf_size);
        sor.filter (*temp_cloud);
    }

    void clear_cloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr& temp_cloud){

        temp_cloud->clear();
        temp_cloud.reset(new pcl::PointCloud<pcl::PointXYZRGB>);
    }

    void publish(pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud, ros::Publisher& pub, std_msgs::Header header, std::string frame_id){

        pcl_conversions::toPCL(header.stamp, cloud->header.stamp);
        cloud->header.seq= header.seq;
        cloud->header.frame_id = frame_id;
        cloud->height = 1;
        cloud->width = cloud->points.size();
        pub.publish(cloud);

    }

    DP icp_process(pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud)
    {


        using namespace PointMatcherSupport;
        pcl::NormalEstimationOMP<pcl::PointXYZRGB, pcl::Normal> ne;
        ne.setInputCloud (cloud);
        pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZRGB> ());
        ne.setSearchMethod (tree);
        pcl::PointCloud<pcl::Normal>::Ptr cloud_normals (new pcl::PointCloud<pcl::Normal>);
        ne.setRadiusSearch (0.05);
        ne.compute (*cloud_normals);

        DP data;

        Eigen::MatrixXf dataNormals(3,cloud_normals->getMatrixXfMap(3,8,0).row(0).size());
        dataNormals.row(0)=cloud_normals->getMatrixXfMap(3,8,0).row(0);
        dataNormals.row(1)=cloud_normals->getMatrixXfMap(3,8,0).row(1);
        dataNormals.row(2)=cloud_normals->getMatrixXfMap(3,8,0).row(2);
        Eigen::MatrixXf datax(1,cloud->getMatrixXfMap(3,8,0).row(0).size());
        Eigen::MatrixXf datay(1,cloud->getMatrixXfMap(3,8,0).row(1).size());
        Eigen::MatrixXf dataz(1,cloud->getMatrixXfMap(3,8,0).row(2).size());
        datax=cloud->getMatrixXfMap(3,8,0).row(0);
        datay=cloud->getMatrixXfMap(3,8,0).row(1);
        dataz=cloud->getMatrixXfMap(3,8,0).row(2);

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


    void mapCb(const sensor_msgs::ImuConstPtr& imu, const sensor_msgs::PointCloud2ConstPtr& ifm)
    {
        textureTrigger=false;
        double currentTime=imu->header.stamp.toSec();

        if (init==false)
        {
            initFn(ifm->header.frame_id);
            startStamp=endStamp=currentTime;
            init=true;
        }


        Eigen::Matrix4d ekfPose=getPoseTf();

        // Eigen::Matrix4d pose=lastEkfPose*lastCorrection*lastEkfPose.inverse()*ekfPose;
        Eigen::Matrix4d pose=ekfPose;

        std::vector<double> planeDefNormal{0,0,1};




        bool rotationSwitch=getRotationSwitch(imu);

        if (rotationSwitch)
        {


            // DP empty;




            if (!lockMap){
                std::cout<< "\033[1;31mRotation detected, mapping is disabled\033[0m\n"<<std::endl;
                lockMap=true;
                endStamp=currentTime;

            }
            else{
                startStamp=currentTime;
                if(localMap->size() > 0){
                    clear_cloud(localMap);
                }
            }

        }

        else
        {


            double insideTh=0.15;  //bounding box sphere, delete inside
            //bounding box sphere, delete outside

            pcl::PCLPointCloud2::Ptr pc2 (new pcl::PCLPointCloud2 ());
            pcl_conversions::toPCL(*ifm,*pc2);
            std::vector<int> indices;
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr temp_cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
            pcl::fromPCLPointCloud2(*pc2,*temp_cloud);
            pcl::removeNaNFromPointCloud(*temp_cloud, *temp_cloud, indices);
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloudIn (new pcl::PointCloud<pcl::PointXYZRGB>);
            *cloudIn= boundingBox(temp_cloud, insideTh, outsideTh);

            *localMap += *cloudIn;

        }


        Eigen::Matrix4d newPose=Eigen::Matrix4d::Identity();

        if (gotMap)
        {
            newPose=prevMapPose*lastCapturedPose.inverse()*capturedPose;

            denoise_map(localMap);
            voxelize(localMap, leaf_size);
            transform_cloud(*localMap, ifmTransform);

            double d=(-planeDefNormal[0]*newPose(0,3)-planeDefNormal[1]*newPose(1,3)-planeDefNormal[2]*(newPose(2,3)));
            std::vector<double> planeDefault{planeDefNormal[0],planeDefNormal[1],planeDefNormal[2],d};
            collapsePoints(*localMap, planeDefault, 0.025);

            localMap->header.frame_id = "base_link";

            if (doIcp)
            {
                DP data=icp_process(localMap);
                //DP mapPointCloud;

                if(globalMap->size() == 0){
                    mapPointCloud=data;
                }

                std::cout<<"map pc has "<<mapPointCloud.getNbPoints()<<" points"<<std::endl;
                std::cout<<"data pc has "<<data.getNbPoints()<<" points"<<std::endl;

                //Eigen::Matrix4d prior=capturedPose*lastCorrection;
                Eigen::Matrix4d prior=newPose;
                Eigen::Matrix4d correction;
                icpFn(data, mapPointCloud, correction, prior, icpParamPath, icpInputParamPath, icpPostParamPath);

                if (!correction.isIdentity()){
                    lastCorrection=prior.inverse()*correction;

                    //capturedPose=correction;
                }
                else
                {
                    lastCorrection=Eigen::Matrix4d::Identity();

                }
                std::cout<<"correction is \n"<<lastCorrection<<std::endl;
                data = rigidTrans->compute(data, prior.cast<float>()*correction.cast<float>());
                mapPointCloud.concatenate(data);

            }

            newPose=prevMapPose*lastCapturedPose.inverse()*capturedPose*lastCorrection;



            // transform_cloud(*localMap, capturedPose*lastCorrection);

        }

        else
        {
            newPose=prevPose*lastEkfPose.inverse()*ekfPose;
        }

        //Eigen::Matrix4d tfCorrection=ekfPose.inverse()*newPose;
        //ekfPose=getPoseTf();  // the robot probably moved while icp was busy
        // updateTf(ifm->header.stamp, ekfPose*tfCorrection*ekfPose.inverse());

        updateTf(ifm->header.stamp, zeroTransform*newPose);
        geometry_msgs::PoseStamped poseStamped;

        poseStamped.header=ifm->header;
        poseStamped.header.frame_id = "map";
        poseStamped.pose.position.x=newPose(0,3);
        poseStamped.pose.position.y=newPose(1,3);
        poseStamped.pose.position.z=newPose(2,3);
        Eigen::Matrix3d dcm;
        dcm=newPose.block(0,0,3,3);
        Eigen::Quaterniond q(dcm);
        poseStamped.pose.orientation.x=q.x();
        poseStamped.pose.orientation.y=q.y();
        poseStamped.pose.orientation.z=q.z();
        poseStamped.pose.orientation.w=q.w();
        posePub.publish(poseStamped);




        path.header=poseStamped.header;
        path.header.frame_id="map";
        path.poses.push_back(poseStamped);
        pathPub.publish(path);



            publish(localMap, localMapPub, lastHeader, "base_link" );
            //publish_colored_texture(texture,newPose,localMap);
            transform_cloud(*localMap, zeroTransform*newPose);


            *globalMap += *localMap;
            publish(globalMap, globalMapPub, lastHeader, "map");
            clear_cloud(localMap);
            prevMapPose=prevPose=newPose;
            lastCapturedPose=capturedPose;




        prevPose=newPose;


        lastEkfPose=ekfPose;

    }
public:
    Mapper() : n("~"), it(n) {

        n.param<bool>("ICP", doIcp, false);
        n.param<bool>("ICP_compute_prob_dynamic", computeProbDynamic, false);
        n.param<double>("max_range", outsideTh, 4);

        //ROS_INFO("Texture trigger srv ready.");

        n.param<float>("leaf_size", leaf_size, 0.01);
        textureResolution=0.01;
        rigidTrans = PM::get().REG(Transformation).create("RigidTransformation");
        startStamp=endStamp=0;
        lockMap=false;

        icpParamPath="/home/gchahine/catkin_ws/src/crawler-mapper-2/icpConfig/icp_param.yaml";
        icpInputParamPath="/home/gchahine/catkin_ws/src/crawler-mapper-2/icpConfig/input_filters.yaml";
        icpPostParamPath="/home/gchahine/catkin_ws/src/crawler-mapper-2/icpConfig/mapPost_filters.yaml";
        //computeProbDynamic=true;
        // doIcp=true;
        useTf=false;
        textureTrigger=false;
        skipRansac=true;
        lastTime=-1;
        iteration=1;
        gx=gy=gz=gxx=gyy=gzz=aa=bb=cc=dd=x=y=n1=n2=n3=0;
        //leaf_size=0.01;


        q0 = tf::createQuaternionFromRPY(0, 0.33, 0);
        rotation.x()=q0[0];
        rotation.y()=q0[1];
        rotation.z()=q0[2];
        rotation.w()=q0[3];
        rotM=rotation.toRotationMatrix();
        ifmTransform=zeroTransform=lastCorrection=Eigen::Matrix4d::Identity();

        //        ifmTransform.block(0,0,3,3) = rotM;
        //        ifmTransform(2,3)=0.14;

        init=false;

        globalMap = pcl::PointCloud<pcl::PointXYZRGB>::Ptr(new pcl::PointCloud<pcl::PointXYZRGB>);

        localMap = pcl::PointCloud<pcl::PointXYZRGB>::Ptr(new pcl::PointCloud<pcl::PointXYZRGB>);

        capturedPose=lastEkfPose=lastCapturedPose=prevPose=prevMapPose=Eigen::Matrix4d::Identity();
        srand (time(NULL));
        ros::Duration(0.5).sleep();
        std::string transport = "raw";
        n.param("transport",transport,transport);

        imu_sub.subscribe(n, "/imu/imu", 1);
        //odom_sub.subscribe(n, "/odom_ekf", 1);
        pc_sub.subscribe(n, "/ifm3d_filter/cloud_out", 1);

        pathPub = n.advertise<nav_msgs::Path > ("path_map", 1);

        globalMapPub = n.advertise<pcl::PointCloud<pcl::PointXYZRGB> > ("global_pc", 1);

        sync.reset(new Sync(MySyncPolicy(10), imu_sub, pc_sub));

        sync->registerCallback(boost::bind(&Mapper::mapCb, this, _1, _2));
    }

};

int main(int argc, char * argv[]){

    ros::init(argc, argv, "bw2_mapper");
    Mapper var;
    ros::spin();

}