#include "ukf.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 2;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = M_PI/8;

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;

  // State dimension
  n_x_ = 5;

  // Augmented state dimension
  n_aug_ = 7;

  // Sigma point spreading parameter
  lambda_ = 3 - n_x_;

  // Weights of sigma points
  weights_ = VectorXd(2*n_aug_+1);

  // the current NIS for radar
  NIS_radar_ = 0;

  // the current NIS for laser
  NIS_laser_ = 0;
}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  if (meas_package.SensorType == LASER)
  {

  }

  else if (meas_package.SensorType == RADAR)
  {

  }
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  //create augmented mean vector
  VectorXd x_aug = VectorXd(n_aug_);

  //create augmented state covariance
  MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);

  //create sigma point matrix
  MatrixXd Xsig_aug = MatrixXd(n_aug_, 2*n_aug_+1);

  //create augmented mean state
  x_aug.head(n_x_) = x_;
  
  //create augmented covariance matrix
  MatrixXd Q = MatrixXd(2, 2);
  Q <<  std_a_*std_a_, 0,
        0, std_yawdd_*std_yawdd_;
  P_aug.bottomRightCorner(2, 2) = Q;
  P_aug.topLeftCorner(5, 5) = P;

  //calculate square root of P
  MatrixXd A = P_.llt().matrixL();

  //set sigma points as columns of matrix Xsig
  Xsig.col(0) = x_aug;     //column 1 = state vector
  for (int i=0; i<n_aug_; i++)
  {
      Xsig_aug.col(i+1) = x_aug + sqrt(lambda + n_aug_) * A.col(i);
      Xsig_aug.col(i+1+n_x) = x_aug - sqrt(lambda + n_aug_) * A.col(i);
  }

  //create matrix with predicted sigma points as columns
  MatrixXd Xsig_pred = MatrixXd(n_x_, 2 * n_aug_ + 1);

  //predict sigma points
  VectorXd x_k = VectorXd(n_aug_);
  for (int i=0; i<2*n_aug_+1; i++)
  {
      //extract values for better readability
      x_k = Xsig_aug.col(i);
      double p_x = x_k(0);
      double p_y = x_k(1);
      double v = x_k(2);
      double yaw = x_k(3);
      double yawd = x_k(4);
      double nu_a = x_k(5);
      double nu_yawdd = x_k(6);

      //avoid division by zero
      if(fabs(yawd) > 0.001)
      {
          Xsig_pred(0, i) = p_x + 
                            ((v/yawd) * (sin(yaw + (yawd*delta_t)) - sin(yaw))) +
                            (0.5 * (delta_t*delta_t) * cos(yaw)*nu_a);
          Xsig_pred(1, i) = p_y + 
                            ((v/yawd) * (-cos(yaw + (yawd*delta_t)) + cos(yaw))) +
                            (0.5 * (delta_t*delta_t) * sin(yaw)*nu_a);
      }
      else
      {
          Xsig_pred(0, i) = p_x + 
                            (v * cos(yaw) * delta_t) + 
                            (0.5 * (delta_t*delta_t) * cos(yaw)*nu_a);
          Xsig_pred(1, i) = p_y + 
                            (v * sin(yaw) * delta_t) +
                            (0.5 * (delta_t*delta_t) * sin(yaw)*nu_a);
      }
      Xsig_pred(2, i) = v + 0 + (delta_t*nu_a);
      Xsig_pred(3, i) = yaw + (yawd*delta_t) + (0.5 * (delta_t*delta_t) * nu_yawdd);
      Xsig_pred(4, i) = yawd + 0 + (delta_t*nu_yawdd);
  }

  //set weights
  weights_(0) = lambda_/(lambda_ + n_aug_);
  for (int i=1; i<2*n_aug_+1; i++)
  {
      weights_(i) = 1/(2*(lambda_ + n_aug_));
  }
  
  //predict state mean
  for (int i=0; i<2*n_aug_+1; i++)
  {
      x_ += weights_(i)*Xsig_pred.col(i);
  }
  
  //predict state covariance matrix
  for (int i=0; i<2*n_aug_+1; i++)
  {
      VectorXd x_diff = Xsig_pred.col(i)-x_;
      while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
      while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;
      P_ += weights_(i)*x_diff*x_diff.transpose();
  }
  
  //set measurement dimension, radar can measure r, phi, and r_dot
  int n_z_ = 3;

  //create matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd(n_z_, 2 * n_aug_ + 1);

  //mean predicted measurement
  VectorXd z_pred = VectorXd(n_z_);
  
  //measurement covariance matrix S
  MatrixXd S_ = MatrixXd(n_z_,n_z_);

  //transform sigma points into measurement space
  for (int i=0; i<2*n_aug_+1; i++)
  {
      double p_x = Xsig_pred(0,i);
      double p_y = Xsig_pred(1,i);
      double yaw = Xsig_pred(3,i);
      double v = Xsig_pred(2,i);
      Zsig(0,i) = sqrt(p_x*p_x + p_y*p_y);
      Zsig(1,i) = atan2(p_y, p_x);
      Zsig(2,i) = ((p_x*cos(yaw)*v) + (p_y*sin(yaw)*v))/sqrt(p_x*p_x + p_y*p_y);
  }
  
  //calculate mean predicted measurement
  for (int i=0; i<2*n_aug_+1; i++)
  {
      z_pred += weights_(i) * Zsig.col(i);
  }
  
  //calculate measurement covariance matrix S
  MatrixXd R_ = MatrixXd(n_z, n_z);

  //measurement noise covariance
  R_ <<  std_radr*std_radr, 0, 0,
        0, std_radphi*std_radphi, 0,
        0, 0, std_radrd*std_radrd;

  for (int i=0; i<2*n_aug_+1; i++)
  {
      VectorXd z_diff = Zsig.col(i) - z_pred;

      //angle normalization
      while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
      while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

      S_ += weights_(i) * z_diff * z_diff.transpose();
  }
  //add measurement noise covariance matrix
  S_ += R_;

}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use lidar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the lidar NIS.
  */
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use radar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the radar NIS.
  */
}
