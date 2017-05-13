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
  std_a_ = 1.0;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.5;

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
  lambda_ = 3 - n_aug_;

  // Weights of sigma points
  weights_ = VectorXd(2*n_aug_+1);

  // the current NIS for radar
  NIS_radar_ = 0;

  // the current NIS for laser
  NIS_laser_ = 0;

  // Initialization check
  is_initialized_ = false;

  //create matrix with predicted sigma points as columns
  Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);



  }

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  if (!is_initialized_)
  {
    previous_timestamp_ = meas_package.timestamp_;

    // Initialize the state and covariance matrix with the first measurement
    float p_x = meas_package.raw_measurements_[0];
    float p_y = meas_package.raw_measurements_[1];
    if (meas_package.sensor_type_ == MeasurementPackage::LASER)
    {
      x_ << p_x, p_y, 0, 0, 0;
    }
    else if (meas_package.sensor_type_ == MeasurementPackage::RADAR)
    {
      float rho = p_x * cos(p_y);
      float phi = p_x * sin(p_y);
      float rho_dot = meas_package.raw_measurements_[2];
      x_ << rho, phi, 0, 0, 0;
    }

    // Initial state covariance matrix P  
    P_ << 1, 0, 0, 0, 0,
          0, 1, 0, 0, 0,
          0, 0, 1000, 0, 0,
          0, 0, 0, 100, 0,
          0, 0, 0, 0, 1;

    is_initialized_ = true;
    return;
  }

  //compute the time elapsed between the current and previous measurements
  double delta_t = (meas_package.timestamp_ - previous_timestamp_) / 1000000.0;
  previous_timestamp_ = meas_package.timestamp_;

  // Call Prediction step
  Prediction(delta_t);

  // Call Update step
  if (use_laser_ && meas_package.sensor_type_ == MeasurementPackage::LASER)
  {
    UpdateLidar(meas_package);
  }
  else if (use_radar_ && meas_package.sensor_type_ == MeasurementPackage::RADAR)
  {
    UpdateRadar(meas_package);
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
  Xsig_aug.fill(0);

  //create augmented mean state
  x_aug.head(n_x_) = x_;
  x_aug(5) = 0;
  x_aug(6) = 0;
  
  //create augmented covariance matrix
  MatrixXd Q = MatrixXd(2, 2);
  Q <<  std_a_*std_a_, 0,
        0, std_yawdd_*std_yawdd_;
  
  P_aug.fill(0.0);
  P_aug.topLeftCorner(n_x_, n_x_) = P_;
  P_aug.bottomRightCorner(2, 2) = Q;

  //create square root matrix
  MatrixXd A = P_aug.llt().matrixL();

  //set sigma points as columns of matrix Xsig
  Xsig_aug.col(0) = x_aug;     //column 1 = state vector
  for (int i=0; i<n_aug_; i++)
  {
    Xsig_aug.col(i+1) = x_aug + sqrt(lambda_ + n_aug_) * A.col(i);
    Xsig_aug.col(i+1+n_aug_) = x_aug - sqrt(lambda_ + n_aug_) * A.col(i);
  } 

  Xsig_pred_.fill(0);

  //predict sigma points
  VectorXd x_k = VectorXd(n_aug_);
  for (int i=0; i<2*n_aug_+1; i++)
  {
    //extract values for better readability
    x_k.fill(0);
    x_k = Xsig_aug.col(i);
    double p_x = x_k(0);
    double p_y = x_k(1);
    double v = x_k(2);
    double yaw = x_k(3);
    double yawd = x_k(4);
    double nu_a = x_k(5);
    double nu_yawdd = x_k(6);

    //predicted state values
    double px_p, py_p;

    //avoid division by zero
    if(fabs(yawd) > 0.001)
    {
      px_p = p_x + v/yawd * ( sin (yaw + yawd*delta_t) - sin(yaw));
      py_p = p_y + v/yawd * ( cos(yaw) - cos(yaw+yawd*delta_t) );
    }
    else
    {
      px_p = p_x + v*delta_t*cos(yaw);
      py_p = p_y + v*delta_t*sin(yaw);
    }

    double v_p = v;
    double yaw_p = yaw + yawd*delta_t;
    double yawd_p = yawd;

    //add noise
    px_p = px_p + 0.5*nu_a*delta_t*delta_t * cos(yaw);
    py_p = py_p + 0.5*nu_a*delta_t*delta_t * sin(yaw);
    v_p = v_p + nu_a*delta_t;

    yaw_p = yaw_p + 0.5*nu_yawdd*delta_t*delta_t;
    yawd_p = yawd_p + nu_yawdd*delta_t;

    //write predicted sigma point into right column
    Xsig_pred_(0,i) = px_p;
    Xsig_pred_(1,i) = py_p;
    Xsig_pred_(2,i) = v_p;
    Xsig_pred_(3,i) = yaw_p;
    Xsig_pred_(4,i) = yawd_p;
  }

  //set weights
  weights_(0) = lambda_/(lambda_ + n_aug_);
  for (int i=1; i<2*n_aug_+1; i++)
  {
    double weight = 0.5/(lambda_ + n_aug_);
    weights_(i) = weight;
  }
  
  //predict state mean
  x_.fill(0.0);
  for (int i=0; i<2*n_aug_+1; i++)
  {
    x_ += weights_(i)*Xsig_pred_.col(i);
  }
  
  //predict state covariance matrix
  P_.fill(0.0);
  for (int i=0; i<2*n_aug_+1; i++)
  {
    VectorXd x_diff = Xsig_pred_.col(i)-x_;
    while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
    while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;
    P_ += weights_(i) * x_diff * x_diff.transpose();
  }
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  //set measurement dimension
  int n_z_ = 2;

  //create matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd(n_z_, 2 * n_aug_ + 1);

  //mean predicted measurement
  VectorXd z_pred = VectorXd(n_z_);
  
  //measurement covariance matrix S
  MatrixXd S_ = MatrixXd(n_z_,n_z_);

  //prediction noise matrix
  MatrixXd R_ = MatrixXd(n_z_, n_z_);

  //cross correlation matrix
  MatrixXd Tc = MatrixXd(n_x_, n_z_);

  //ground truth measurement
  VectorXd z = meas_package.raw_measurements_;

  //transform sigma points into measurement space
  for (int i=0; i<2*n_aug_+1; i++)
  {
    double p_x = Xsig_pred_(0,i);
    double p_y = Xsig_pred_(1,i);
    Zsig.col(i) << p_x, p_y;
  }

  //calculate mean predicted measurement
  z_pred.fill(0.0);
  for (int i=0; i<2*n_aug_+1; i++)
  {
    z_pred += weights_(i) * Zsig.col(i);
  }

  //calculate measurement covariance matrix S
  S_.fill(0.0);
  for (int i=0; i<2*n_aug_+1; i++)
  {
    VectorXd z_diff = Zsig.col(i) - z_pred;
    S_ += weights_(i) * z_diff * z_diff.transpose();
  }

  //measurement noise covariance
  R_ <<  std_laspx_*std_laspx_, 0,
         0, std_laspy_*std_laspy_;

  //add measurement noise covariance matrix
  S_ += R_;

  //calculate cross correlation matrix
  Tc.fill(0);
  for (int i=0; i<2 * n_aug_ + 1; i++)
  {
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    
    VectorXd z_diff = Zsig.col(i) - z_pred;
    Tc += weights_(i) * x_diff * z_diff.transpose();
  }

  //calculate Kalman gain K
  MatrixXd K = Tc * S_.inverse();
  
  //update state mean and covariance matrix
  VectorXd z_diff_ = z - z_pred;
  x_ += K*(z_diff_);
  P_ -= K * S_ * K.transpose();

  //calculate NIS
  NIS_laser_ = z_diff_.transpose() * S_.inverse() * z_diff_;
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  //set measurement dimension, radar can measure r, phi, and r_dot
  int n_z_ = 3;

  //create matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd(n_z_, 2 * n_aug_ + 1);

  //mean predicted measurement
  VectorXd z_pred = VectorXd(n_z_);
  
  //measurement covariance matrix S
  MatrixXd S_ = MatrixXd(n_z_,n_z_);

  //prediction noise matrix
  MatrixXd R_ = MatrixXd(n_z_, n_z_);

  //cross correlation matrix
  MatrixXd Tc = MatrixXd(n_x_, n_z_);

  //ground truth measurement
  VectorXd z = meas_package.raw_measurements_;

  //transform sigma points into measurement space
  for (int i=0; i<2*n_aug_+1; i++)
  {
    double p_x = Xsig_pred_(0,i);
    double p_y = Xsig_pred_(1,i);
    double v = Xsig_pred_(2,i);
    double yaw = Xsig_pred_(3,i);
    double rho = sqrt((p_x*p_x) + (p_y*p_y));
    double phi = 0.0;
    double rho_dot = 0.0;

    //check for division by 0
    if (fabs(p_x) > 0.001)
    {
      phi = atan2(p_y, p_x);
    }

    if (fabs(rho) > 0.001)
    {
      rho_dot = (p_x * cos(yaw) * v + p_y * sin(yaw) * v) / rho;
    }
    
    Zsig(0,i) = rho;
    Zsig(1,i) = phi;
    Zsig(2,i) = rho_dot;
  }

  //calculate mean predicted measurement
  z_pred.fill(0.0);
  for (int i=0; i<2*n_aug_+1; i++)
  {
    z_pred += weights_(i) * Zsig.col(i);
  }
  
  //calculate measurement covariance matrix S
  S_.fill(0.0);

  for (int i=0; i<2*n_aug_+1; i++)
  {
    VectorXd z_diff = Zsig.col(i) - z_pred;

    //angle normalization
    while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
    while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

    S_ += weights_(i) * z_diff * z_diff.transpose();
  }

  //measurement noise covariance
  R_.fill(0);
  R_ <<  std_radr_*std_radr_, 0, 0,
         0, std_radphi_*std_radphi_, 0,
         0, 0, std_radrd_*std_radrd_;

  //add measurement noise covariance matrix
  S_ += R_;

  //calculate cross correlation matrix
  Tc.fill(0.0);
  for (int i=0; i<2 * n_aug_ + 1; i++)
  {
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    
    //angle normalization
    while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
    while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;
    
    VectorXd z_diff = Zsig.col(i) - z_pred;
    
    //angle normalization
    while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
    while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

    Tc += weights_(i) * x_diff * z_diff.transpose();
  }

  //calculate Kalman gain K;
  MatrixXd K = Tc * S_.inverse();

  VectorXd z_diff = z - z_pred;

  //angle normalization
  while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
  while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;
  
  //update state mean and covariance matrix
  x_ += K * z_diff;
  P_ -= K * S_ * K.transpose();

  //calculate NIS
  NIS_radar_ = z_diff.transpose() * S_.inverse() * z_diff;
}
