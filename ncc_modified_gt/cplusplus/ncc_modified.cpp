#include "mcl.h"

using namespace std;
using namespace cv;
mcl::mcl()
{
  m_sync_count = 0;
  gen.seed(rd()); //Set random seed for random engine
  Mat temp;
  
  // gridMap = cv::imread("/home/mav-lab/slam_ws/src/mcl_2d_lidar_ros/global_ortho.png",cv::IMREAD_GRAYSCALE); //Original gridmap (For show)
  temp = cv::imread("/home/mav-lab/slam_ws/src/mcl_2d_lidar_ros/global_ortho.png",cv::IMREAD_GRAYSCALE); //grdiamp for use.
  gridMapCV = tool::cvResizeMat(temp, 1 / 3.54);
  cout<< gridMapCV.cols << " " << gridMapCV.rows << endl;

  //--YOU CAN CHANGE THIS PARAMETERS BY YOURSELF--//
  numOfParticle = 1500; // Number of Particles.
  minOdomDistance = 0.05; // [m]
  minOdomAngle = 30; // [deg]
  repropagateCountNeeded = 1; // [num]
  odomCovariance[0] = 0.05; // Rotation to Rotation
  odomCovariance[1] = 0.05; // Translation to Rotation
  odomCovariance[2] = 0.05; // Translation to Translation
  odomCovariance[3] = 0.05; // Rotation to Translation
  odomCovariance[4] = 0.10; // X
  odomCovariance[5] = 0.10; // Y
  template_size = 150; // Template(square) size
  init_angle = -93.2; // Rotation init guess [degree]

  //--DO NOT TOUCH THIS PARAMETERS--//
  imageResolution = 0.1; // [m] per [pixel]
  tf_laser2robot << 1,0,0,0.0,
                    0,1,0,0,
                    0,0,1,0,
                    0,0,0,1; // TF (laser frame to robot frame)
  mapCenterX = round(gridMapCV.cols/2) * imageResolution; // [m]
  mapCenterY = round(gridMapCV.rows/2) * imageResolution; // [m]
  isOdomInitialized = false; //Will be true when first data incoming.
  predictionCounter = 0;

  initializeParticles(); // Initialize particles.
  showInMap();
}

mcl::~mcl()
{

}

/* INITIALIZE PARTICLES UNIFORMLY TO THE MAP
 */
void mcl::initializeParticles()
{
  particles.clear();
  std::uniform_real_distribution<float> x_pos(1,
                                              gridMapCV.cols * imageResolution / 2);
  std::uniform_real_distribution<float> y_pos(1,
                                              gridMapCV.rows * imageResolution / 4); //heuristic setting. (to put particles into the map)
  // std::uniform_real_distribution<float> theta_pos(-M_PI,M_PI); // -180 ~ 180 Deg
  //SET PARTICLES BY RANDOM DISTRIBUTION
  for(int i=0;i<numOfParticle;i++)
  {
    particle particle_temp;
    float randomX = x_pos(gen);
    float randomY = y_pos(gen);
    // float randomTheta = theta_pos(gen);
    particle_temp.pose = tool::xyzrpy2eigen(randomX,randomY,0,0,0,init_angle/180);
    particle_temp.score = 1 / (double)numOfParticle;
    particle_temp.theta = init_angle;
    particles.push_back(particle_temp);
  }
  showInMap();
}

void mcl::prediction(Eigen::Matrix4f diffPose)
{
  std::cout<<"Predicting..."<<m_sync_count<<std::endl;
  Eigen::VectorXf diff_xyzrpy = tool::eigen2xyzrpy(diffPose); // {x,y,z,roll,pitch,yaw} (z,roll,pitch assume to 0)

  /* Your work.
   * Input : diffPose,diff_xyzrpy (difference of odometry pose).
   * To do : update(propagate) particle's pose.
   */

  //------------  FROM HERE   ------------------//
  //// Using odometry model
  double delta_trans = sqrt(pow(diff_xyzrpy(0), 2)+ pow(diff_xyzrpy(1),2));
  double delta_rot1 = atan2(diff_xyzrpy(1), diff_xyzrpy(0));
  double delta_rot2 = diff_xyzrpy(5) - delta_rot1;

  std::default_random_engine generator;
  if(delta_rot1  > M_PI)
          delta_rot1 -= (2*M_PI);
  if(delta_rot1  < -M_PI)
          delta_rot1 += (2*M_PI);
  if(delta_rot2  > M_PI)
          delta_rot2 -= (2*M_PI);
  if(delta_rot2  < -M_PI)
          delta_rot2 += (2*M_PI);
  //// Add noises to trans/rot1/rot2
  double trans_noise_coeff = odomCovariance[2]*fabs(delta_trans) + odomCovariance[3]*fabs(delta_rot1+delta_rot2);
  double rot1_noise_coeff = odomCovariance[0]*fabs(delta_rot1) + odomCovariance[1]*fabs(delta_trans);
  double rot2_noise_coeff = odomCovariance[0]*fabs(delta_rot2) + odomCovariance[1]*fabs(delta_trans);


  float scoreSum = 0;
  for(int i=0;i<particles.size();i++)
  {
    std::normal_distribution<double> gaussian_distribution(0, 1);

    delta_trans = delta_trans + gaussian_distribution(gen) * trans_noise_coeff;
    // delta_rot1 = delta_rot1 + gaussian_distribution(gen) * rot1_noise_coeff;
    // delta_rot2 = delta_rot2 + gaussian_distribution(gen) * rot2_noise_coeff;

    // double x = delta_trans * cos(delta_rot1) + gaussian_distribution(gen) * odomCovariance[4];
    // double y = delta_trans * sin(delta_rot1) + gaussian_distribution(gen) * odomCovariance[5];

    double x = delta_trans + gaussian_distribution(gen) * odomCovariance[4];
    double y = delta_trans + gaussian_distribution(gen) * odomCovariance[5];
    // double theta = delta_rot1 + delta_rot2 + gaussian_distribution(gen) * odomCovariance[0]*(M_PI/180.0);

    Eigen::Matrix4f diff_odom_w_noise = tool::xyzrpy2eigen(x, y, 0, 0, 0, 0);

    Eigen::Matrix4f pose_t_plus_1 = particles.at(i).pose * diff_odom_w_noise;

    // cout << " delta_transx = " << x << " delta_transy = " << y 
    //     << " pose_t_plus_1_x = " << pose_t_plus_1(0,3) << " pose_t_plus_1_y = " << pose_t_plus_1(1,3) << endl;


    ////For debugging
//    Eigen::Matrix4f pose_t_plus_1 = particles.at(i).pose * diffPose;
    scoreSum = scoreSum + particles.at(i).score; // For normalization
    particles.at(i).pose= pose_t_plus_1;
  }

  //------------  TO HERE   ------------------//

  for(int i=0;i<particles.size();i++)
  {
    particles.at(i).score = particles.at(i).score/scoreSum; // normalize the score
  }

  showInMap();

}

void mcl::weightning(Eigen::Matrix4Xf laser)
{
  float maxScore = 0;
  float scoreSum = 0;

  /* Your work.
   * Input : laser measurement data
   * To do : update particle's weight(score)
   */

  for(int i=0;i<particles.size();i++)
  {
    //Todo : Transform laser data into global frame to map matching
    //Input : laser (4 x N matrix of laser points in lidar sensor's frame)
    //        particles.at(i).pose (4 x 4 matrix of robot pose)
    //        tf_laser2robot (4 x 4 matrix of transformatino between robot and sensor)
    //Output : transLaser (4 x N matrix of laser points in global frame)

    Eigen::Matrix4Xf transLaser = particles.at(i).pose* tf_laser2robot* laser; // now this is lidar sensor's frame.

    //--------------------------------------------------------//

    float calcedWeight = 0;

    for(int j=0;j<transLaser.cols();j++)
    {
      //TODO :  translate each laser point (in [m]) to pixel frame.  (transLaser(0,i) 's unit is [m]) (You will use it in MCL too! remember!)
      //Input :  transLaser(0,j), transLaser(1,j)  (laser point's pose in global frame)
      //         imageResolution
      //         gridMap.rows , gridMap.cols (size of image)
      //         mapCenterX, mapCenterY (center of map's position)
      //Output : ptX, ptY (laser point's pixel position)

      int ptX  = static_cast<int>((transLaser(0, j) - mapCenterX + (300.0*imageResolution)/2)/imageResolution);
      int ptY = static_cast<int>((transLaser(1, j) - mapCenterY + (300.0*imageResolution)/2)/imageResolution);

      //----------------------------------------------------------------------------------------//

      if(ptX<0 || ptX>=gridMapCV.cols || ptY<0 ||  ptY>=gridMapCV.rows) continue; // dismiss if the laser point is at the outside of the map.
      else
      {
        double img_val =  gridMapCV.at<uchar>(ptY,ptX)/(double)255; //calculate the score.
        calcedWeight += img_val; //sum up the score.
      }


    }
    particles.at(i).score = particles.at(i).score + (calcedWeight / transLaser.cols()); //Adding score to particle.
    scoreSum += particles.at(i).score;
    if(maxScore < particles.at(i).score) // To check which particle has max score
    {
      maxProbParticle = particles.at(i);
      maxProbParticle.scan = laser;
      maxScore = particles.at(i).score;
    }
  }
  for(int i=0;i<particles.size();i++)
  {
    particles.at(i).score = particles.at(i).score/scoreSum; // normalize the score
  }
}

void mcl::resampling()
{
  std::cout<<"Resampling..."<<m_sync_count<<std::endl;

  //Make score line (roullette)
  std::vector<double> particleScores;
  std::vector<particle> particleSampled;
  double scoreBaseline = 0;
  for(int i=0;i<particles.size();i++)
  {
    scoreBaseline += particles.at(i).score;
    particleScores.push_back(scoreBaseline);
  }

  std::uniform_real_distribution<double> dart(0, scoreBaseline);
  for(int i=0;i<particles.size();i++)
  {
    double darted = dart(gen); //darted number. (0 to maximum scores)
    auto lowerBound = std::lower_bound(particleScores.begin(), particleScores.end(), darted);
    int particleIndex = lowerBound - particleScores.begin(); // Index of particle in particles.

    //TODO : put selected particle to array 'particleSampled' with score reset.

    particle selectedParticle = particles.at(particleIndex); // Which one you have to select?

    particleSampled.push_back(selectedParticle);

  }
  particles = particleSampled;
}


//DRAWING FUNCTION.
void mcl::showInMap()
{
//  cv::Mat showMap(gridMap.cols, gridMap.rows, CV_8UC3);
  cv::Mat showMap;
  cv::cvtColor(gridMapCV, showMap, cv::COLOR_GRAY2BGR);

  for(int i=0;i<numOfParticle;i++)
  {
    //Todo : Convert robot pose in image frame
    //Input :  particles[i].pose(0,3), particles[i].pose(1,3) (x,y position in [m])
    //         imageResolution
    //         gridMap.rows , gridMap.cols (size of image)
    //         mapCenterX, mapCenterY (center of map's position)
    //Output : xPos, yPos (pose in pixel value)

    int xPos  = static_cast<int>((particles.at(i).pose(0, 3)) / imageResolution + template_size / 2);
    int yPos = static_cast<int>((particles.at(i).pose(1, 3)) / imageResolution + template_size / 2);

    //---------------------------------------------//
    cv::circle(showMap,cv::Point(xPos,yPos),1,cv::Scalar(255,0,0),-1);
  }
  if(maxProbParticle.score > 0)
  {
    //Todo : Convert robot pose in image frame
    //Input :  maxProbParticle.pose(0,3), maxProbParticle.pose(1,3) (x,y position in [m])
    //         imageResolution
    //         gridMap.rows , gridMap.cols (size of image)
    //         mapCenterX, mapCenterY (center of map's position)
    //Output : xPos, yPos (pose in pixel value)

    //// Original
//    int xPos = static_cast<int>((maxProbParticle.pose(0, 3) - mapCenterX + (300.0*imageResolution)/2)/imageResolution);
//    int yPos = static_cast<int>((maxProbParticle.pose(1, 3) - mapCenterY + (300.0*imageResolution)/2)/imageResolution);

    //// Estimate position using all particles
    float x_all = 0;
    float y_all = 0;
    for(int i=0;i<particles.size();i++)
    {
      x_all = x_all + particles.at(i).pose(0,3) * particles.at(i).score;
      y_all = y_all + particles.at(i).pose(1,3) * particles.at(i).score;
    }
    int xPos = static_cast<int>((x_all - mapCenterX)/imageResolution + template_size / 2);
    int yPos = static_cast<int>((y_all - mapCenterY)/imageResolution + template_size / 2);


    //---------------------------------------------//

    cv::circle(showMap,cv::Point(xPos,yPos),2,cv::Scalar(0,0,255),-1);

  }
  cv::imshow("MCL2", showMap);
  cv::waitKey(1);
}

void mcl::updateLaserData(Eigen::Matrix4f pose, Eigen::Matrix4Xf laser)
{
  if(!isOdomInitialized)
  {
    odomBefore = pose; // Odom used at last prediction.
    isOdomInitialized = true;
  }
  //When difference of odom from last correction is over some threshold, Doing correction.

  //Calculate difference of distance and angle.

  Eigen::Matrix4f diffOdom = odomBefore.inverse()*pose; // odom after = odom New * diffOdom
  Eigen::VectorXf diffxyzrpy = tool::eigen2xyzrpy(diffOdom); // {x,y,z,roll,pitch,yaw}
  float diffDistance = sqrt(pow(diffxyzrpy[0],2)+pow(diffxyzrpy[1],2));
  float diffAngle = fabs(diffxyzrpy[5])*180.0/3.141592;

  if(diffDistance>minOdomDistance || diffAngle>minOdomAngle)
  {
    //Doing correction & prediction
    cout << "Predicting step" << endl;
    prediction(diffOdom);
        
    cout << "Weightning step" << endl;
    weightning(laser);

    predictionCounter++;
    if(predictionCounter == repropagateCountNeeded)
    {
      resampling();
      predictionCounter = 0;
    }

    m_sync_count = m_sync_count + 1;
    odomBefore = pose;
  }
}

void mcl::weightning_NCC(cv::Mat template_image)
{
  float maxScore = 0;
  float scoreSum = 0;

  for(int i=0;i<particles.size();i++)
  {

    Eigen::Matrix4Xf particle_pose = particles.at(i).pose;
    float particle_rot = particles.at(i).theta;
    cv::Mat image;

    image = tool::cvRotateMat(template_image, particle_rot);
    cv::imshow("template", image);
    cv::waitKey(1);
    cv::Rect rect(int(particle_pose(0,3) / imageResolution), int(particle_pose(1,3) / imageResolution), image.rows, image.cols);
    // cout << "particle_pose(0,3) = " << particle_pose(0,3) / imageResolution << " particle_pose(1,3) = " << particle_pose(1,3) / imageResolution << " image.rows = " << image.rows << " image.cols = " << image.cols << endl;
    cv::Mat global_roi = gridMapCV(rect);
    cv::imshow("particle view", global_roi);
    cv::waitKey(1);

    double sum_img = 0.0;
    double sum_temp = 0.0;
    double sum_2 = 0.0;
    double calcedWeight = 0.0;

    // NCC
    for(int j=0;j<image.cols;j++)
    {
      for(int k=0;k<image.rows;k++)
      {
        // int ptX = static_cast<int>(j + particle_pose(0,3) / imageResolution);
        // int ptY = static_cast<int>(k + particle_pose(1,3) / imageResolution);
        // cout << " ptX = " << ptX << " ptY = " << ptY << " particle_posex = " << particle_pose(0,3) << " particle_posey = " << particle_pose(1,3) << " image.rows = " << image.rows << endl; 
        if(image.at<uchar>(j,k) != 255 && global_roi.at<uchar>(j,k) != 0)
        {
          sum_img += pow((global_roi.at<uchar>(j,k)),2); //calculate the score.
          sum_temp += pow(image.at<uchar>(j,k),2);
          sum_2 += (2 - abs((image.cols / 2) - j) / (image.cols / 2) * abs((image.rows / 2) - j)/(image.rows / 2))
                    * global_roi.at<uchar>(particle_pose(0,3) / imageResolution + j, particle_pose(1,3) / imageResolution + k) * image.at<uchar>(j,k);
          // cout << " sum_temp = " << sum_temp << " sum_img = " << sum_img << " sum_2 = " << sum_2 << " calcedWeight = " << calcedWeight << endl; 
        }
      }
    }
    
    calcedWeight = sum_2 / sqrt(sum_img * sum_temp);

    cout << " calcedWeight = " << calcedWeight << endl;
    particles.at(i).score = particles.at(i).score + calcedWeight; //Adding score to particle.
    scoreSum += particles.at(i).score;
    if(maxScore < particles.at(i).score) // To check which particle has max score
    {
      maxProbParticle = particles.at(i);
      maxProbParticle.local_measurement = image;
      maxScore = particles.at(i).score;
    }
  }
  for(int i=0;i<particles.size();i++)
  {
    particles.at(i).score = particles.at(i).score/scoreSum; // normalize the score
  }
}

void mcl::updateImageData(Eigen::Matrix4f pose, cv::Mat local_measurement)
{
  if(!isOdomInitialized)
  {
    odomBefore = pose; // Odom used at last prediction.
    isOdomInitialized = true;
  }
  Eigen::Matrix4f diffOdom = odomBefore.inverse() * pose; // odom after = odom New * diffOdom
  Eigen::VectorXf diffxyzrpy = tool::eigen2xyzrpy(diffOdom); // {x,y,z,roll,pitch,yaw}
  float diffDistance = sqrt(pow(diffxyzrpy[0],2) + pow(diffxyzrpy[1],2));
  float diffAngle = fabs(diffxyzrpy[5]) * 180.0 / 3.141592;

  if(diffDistance>minOdomDistance || diffAngle>minOdomAngle)
  {
    //Doing correction & prediction
    cout << "Predicting step" << endl;
    prediction(diffOdom);
    
    cout << "Weightning_NCC step" << endl;
    // cv::imshow("test", local_measurement);
    // cv::waitKey();
    weightning_NCC(local_measurement);

    predictionCounter++;
    if(predictionCounter == repropagateCountNeeded)
    {
      cout << "Resampling step" << endl;
      resampling();
      predictionCounter = 0;
    }

    m_sync_count = m_sync_count + 1;
    odomBefore = pose;
  }
}
