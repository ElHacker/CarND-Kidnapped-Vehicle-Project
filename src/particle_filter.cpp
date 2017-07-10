/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

// Random generator to be used globally.
static default_random_engine gen;
static const double YAW_RATE_ZERO_THRESHOLD = 0.00001;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  // TODO: Set the number of particles. Initialize all particles to first
  // position (based on estimates of x, y, theta and their uncertainties from
  // GPS) and all weights to 1. Add random Gaussian noise to each particle.
  // NOTE: Consult particle_filter.h for more information about this method
  // (and others in this file).
  num_particles = 100;

  normal_distribution<double> N_x_init(0, std[0]);
  normal_distribution<double> N_y_init(0, std[1]);
  normal_distribution<double> N_theta_init(0, std[2]);

  for (int i = 0; i < num_particles; i++) {
    Particle p;
    p.id = i;
    p.x = x;
    p.y = y;
    p.theta = theta;
    p.weight = 1.0;

    // Add noise.
    p.x += N_x_init(gen);
    p.y += N_y_init(gen);
    p.theta += N_theta_init(gen);

    // Store the generated particle.
    particles.push_back(p);
  }

  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[],
    double velocity, double yaw_rate) {
	// Add measurements to each particle and add random Gaussian noise.
  // NOTE: When adding noise you may find std::normal_distribution and
  // std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

  // Will be used to add random Gaussian noise
  normal_distribution<double> N_x(0, std_pos[0]);
  normal_distribution<double> N_y(0, std_pos[1]);
  normal_distribution<double> N_theta(0, std_pos[2]);

  for (int i = 0; i < num_particles; i++) {
    double current_theta = particles[i].theta;
    if (fabs(yaw_rate) < YAW_RATE_ZERO_THRESHOLD) {
      particles[i].x += velocity * delta_t * cos(current_theta);
      particles[i].y += velocity * delta_t * sin(current_theta);
    } else {
      particles[i].x += velocity / yaw_rate *
        (sin(current_theta + yaw_rate) + yaw_rate * delta_t) -
          sin(current_theta);
      particles[i].y += velocity / yaw_rate * cos(current_theta) -
        cos(current_theta + yaw_rate * delta_t);
      particles[i].theta += yaw_rate * delta_t;
    }

    // Add noise to particles.
    particles[i].x += N_x(gen);
    particles[i].y += N_y(gen);
    particles[i].theta += N_theta(gen);
  }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted,
    std::vector<LandmarkObs>& observations) {
  // Find the predicted measurement that is closest to each observed
  // measurement and assign the observed measurement to this particular
  // landmark.
  // NOTE: this method will NOT be called by the grading code. But you will
  // probably find it useful to implement this method and use it as a helper
  // during the updateWeights phase.

  for (int i = 0; i < observations.size(); i++) {
    LandmarkObs current_observation = observations[i];

    double min_distance = numeric_limits<double>::max();

    // Id of Landmark from map to be associated.
    int map_id = -1;

    for (int j = 0; j < predicted.size(); j++) {
      LandmarkObs current_prediction = predicted[j];

      // Calculate distance between current and predicted landmarks.
      double distance = dist(current_observation.x, current_observation.y,
          current_prediction.x, current_prediction.y);

      // Select the nearest landmark to the predicted one.
      if (distance < min_distance) {
        min_distance = distance;
        map_id = current_prediction.id;
      }
    }

    // Update the observations with the nearest landmark.
    observations[i].id = map_id;
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
		std::vector<LandmarkObs> observations, Map map_landmarks) {
  // TODO: Update the weights of each particle using a mult-variate Gaussian
  // distribution. You can read more about this distribution here:
  // https://en.wikipedia.org/wiki/Multivariate_normal_distribution
  // NOTE: The observations are given in the VEHICLE'S coordinate system. Your
  // particles are located according to the MAP'S coordinate system. You will
  // need to transform between the two systems.s
  // Keep in mind that this transformation requires both rotation AND
  // translation (but no scaling).
	// The following is a good resource for the theory:
	// https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
  // and the following is a good resource for the actual equation to implement
  // (look at equation 3.33)
	// http://planning.cs.uiuc.edu/node99.html
}

void ParticleFilter::resample() {
  // TODO: Resample particles with replacement with probability proportional to
  // their weight.
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

}

Particle ParticleFilter::SetAssociations(Particle particle,
    std::vector<int> associations, std::vector<double> sense_x,
    std::vector<double> sense_y) {
  // particle: the particle to assign each listed association, and association's
  // (x,y) world coordinates mapping to associations: The landmark id that goes
  // along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best) {
	vector<int> v = best.associations;
	stringstream ss;
  copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseX(Particle best) {
	vector<double> v = best.sense_x;
	stringstream ss;
  copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseY(Particle best) {
	vector<double> v = best.sense_y;
	stringstream ss;
  copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}
