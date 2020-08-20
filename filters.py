import numpy as np
import cv2
import videohelper

class KalmanFilter(object):
    
    """Kalman Filter"""

    def __init__(self, init_x, init_y, Q=0.1 * np.eye(4), R=0.1 * np.eye(2)):
        # State
        self.state = np.array([init_x, init_y, 5, 0])
        # Covariance
        self.P = np.array([[0.1, 0, 0, 0], [0, 0.1, 0, 0], [0, 0, 0.1, 0], [0, 0, 0, 0.1]], dtype=np.float)
        # Time stemp
        self.dt = 1.0
        # System dynamics
        self._Dt = np.array([[1, 0, self.dt, 0], [0, 1, 0, self.dt], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float)
        # Measurement model
        self._Mt = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype = np.float)
        self._Q = Q
        self._R = R  

    def predict(self):
        # Time-update mean
        self.state = self._Dt @ self.state
        # Time-update covariance
        self.P = self._Dt @ self.P @ self._Dt.T + self._Q

    def correct(self, meas_x, meas_y):
        # Compute Kalman gain
        Kgain = self.P @ (self._Mt.T) @ np.linalg.inv((self._Mt @ self.P @ self._Mt.T + self._R))
        # Measurement update
        mstate = np.array([meas_x, meas_y], dtype=np.float)
        self.state = self.state + Kgain @ (mstate - self._Mt @ self.state)
        self.P = (np.eye(4) - Kgain @ self._Mt) @ self.P

    def process(self, measurement_x, measurement_y):

        self.predict()
        self.correct(measurement_x, measurement_y)

        return self.state[0], self.state[1]


class ParticleFilter(object):
    """A particle filter tracker.
    """

    def __init__(self, frame, template, **kwargs):
        """Initializes the particle filter object.
        """
        self.num_particles = kwargs.get('num_particles')
        self.sigma_exp = kwargs.get('sigma_exp')
        self.sigma_dyn = kwargs.get('sigma_dyn')
        self.template_rect = kwargs.get('template_coords')
        
        self.mse_method = kwargs.get('mse_method', 'bgr')
        self.template = template
        self.frame = frame

        # Sample n number of particles uniformly distributed around the template
        offset = 40

        center_x = int(kwargs.get('template_coords')['x'] + kwargs.get('template_coords')['w'] // 2)
        center_y = int(kwargs.get('template_coords')['y'] + kwargs.get('template_coords')['h'] // 2)

        rand_x = np.random.uniform(center_x - offset, center_x + offset, self.num_particles)
        rand_y = np.random.uniform(center_y - offset, center_y + offset, self.num_particles)

        self.particles = np.vstack((rand_x, rand_y))
        self.particles = self.particles.T
        
        p_weight = 1.0 / self.num_particles
        self.sigma_dyn = self.sigma_dyn
        self.weights = np.ones((self.num_particles)) * p_weight 
        out_path = kwargs["video_mode"]
        self.video_out = videohelper.mp4_video_writer(out_path, (self.template.shape[1], self.template.shape[0]), 20)


    def get_particles(self):
        """Returns the current particles state.
        """
        return self.particles

    def get_weights(self):
        """Returns the current particle filter's weights.
        """
        return self.weights

    def get_error_metric(self, template, frame_cutout):
        """Returns the error metric used based on the similarity measure.
        """

        if (self.mse_method == 'gray'):
            # Convert to gray scales if the input is in BGR
            if (template.shape[2] != 1):
                template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
            if (frame_cutout.shape[2] != 1):
                frame_cutout_gray = cv2.cvtColor(frame_cutout, cv2.COLOR_BGR2GRAY)
            
            # compute divison factor
            m = template_gray.shape[0]
            n = template_gray.shape[1]
            norm_factor = 1.0 / (m * n)

            diff = cv2.subtract(template_gray, frame_cutout_gray)
            squared_diff = cv2.pow(diff, 2)
            squared_diff_sum = np.sum(squared_diff)
            mean_squared_error = squared_diff_sum * norm_factor
        else:
            blue_channel_template = template[:, :, 0]
            green_channel_template = template[:, :, 1]
            red_channel_template = template[:, :, 2]

            blue_channel_frame = frame_cutout[:, :, 0]
            green_channel_frame = frame_cutout[:, :, 1]
            red_channel_frame = frame_cutout[:, :, 2]

            luma_template = 0.3 * red_channel_template + 0.58 * green_channel_template + 0.12 * blue_channel_template
            luma_frame = 0.3 * red_channel_frame + 0.58 * green_channel_frame + 0.12 * blue_channel_frame

            # Compute division factor
            m = template.shape[0]
            n = template.shape[1]
            norm_factor = 1.0 / (m * n)

            diff = luma_template - luma_frame
            squared_diff = diff**2
            sum_squared_diff = np.sum(squared_diff)
            mean_squared_error = sum_squared_diff * norm_factor

        return mean_squared_error

    def resample_particles(self):
        """Returns a new set of particles
        """
        
        # Create an array of indices of each particles
        particle_indices = np.arange(0, self.num_particles)
        resampled_particle_indices = np.random.choice(particle_indices, self.num_particles, replace=True, p=self.weights)

        resampled_partciles = np.zeros(self.particles.shape)
        for i in range(0, len(resampled_particle_indices)):
            resampled_partciles[i, 0] = self.particles[resampled_particle_indices[i], 0]
            resampled_partciles[i, 1] = self.particles[resampled_particle_indices[i], 1]

        return resampled_partciles

    def process(self, frame):
        """Processes a video frame (image) and updates the filter's state.
        """
        norm = 0

        for i in range(0, self.num_particles):
            try:
                # Update particles using motion model
                self.particles[i, 0] = self.particles[i, 0] + np.random.normal(0, self.sigma_dyn, 1)
                self.particles[i, 1] = self.particles[i, 1] + np.random.normal(0, self.sigma_dyn, 1)
                
                # p(z | x)
                lower_row = int(self.particles[i, 1] - self.template.shape[0] // 2)
                upper_row = int(self.particles[i, 1] + self.template.shape[0] // 2)

                lower_column = int(self.particles[i, 0] - self.template.shape[1] // 2)
                upper_column = int(self.particles[i, 0] + self.template.shape[1] // 2)
                frame_cutout = frame[lower_row:upper_row, lower_column:upper_column]

                # Correct the dimension
                frame_cutout = cv2.resize(frame_cutout, (self.template.shape[1], self.template.shape[0]), interpolation=cv2.INTER_CUBIC)
                particle_mse = self.get_error_metric(self.template, frame_cutout)
                self.weights[i] = np.exp(-particle_mse/(2*self.sigma_exp**2))

                # Update normalization factor
                norm += self.weights[i]
            except:
                self.weights[i] = 0
                continue

        # Normalize weights
        try:
            self.weights = self.weights / norm
            
        except:
            pass
    
        # Resample particles
        self.particles = self.resample_particles()
        self.frame = frame
        
    def render(self, frame_in):
        
        """Visualizes current particle filter state.
        """
        color = (0, 0, 255)
        radius = 1
        thickness = 1
        for i in range(self.num_particles):
            cv2.circle(frame_in, (int(self.particles[i, 0]), int(self.particles[i, 1])), radius, color, thickness)

        x_weighted_mean = 0
        y_weighted_mean = 0

        for i in range(self.num_particles):
           x_weighted_mean += self.particles[i, 0] * self.weights[i]
           y_weighted_mean += self.particles[i, 1] * self.weights[i]
        
        try:
            top_left = (int(x_weighted_mean - self.template.shape[1] // 2), int(y_weighted_mean - self.template.shape[0] // 2))
            bottom_right = (int(x_weighted_mean + self.template.shape[1] // 2), int(y_weighted_mean + self.template.shape[0] // 2))
            color = (0, 255, 0)
            cv2.rectangle(frame_in, top_left, bottom_right, color, thickness)
        except:
            pass
        return frame_in


    def video_releaser():
        self.video_out.release()

class AppearanceModelPF(ParticleFilter):
    """A variation of particle filter tracker."""

    def __init__(self, frame, template, **kwargs):

        super(AppearanceModelPF, self).__init__(frame, template, **kwargs)

        self.alpha = kwargs.get('alpha', 0.1)
        
    def process(self, frame):
        """Processes a video frame (image) and updates the filter's state.
        """
        best_particle = np.zeros(2)

        x_weighted_mean = 0
        y_weighted_mean = 0

        for i in range(self.num_particles):
           x_weighted_mean += self.particles[i, 0] * self.weights[i]
           y_weighted_mean += self.particles[i, 1] * self.weights[i]

        best_particle[0] = x_weighted_mean
        best_particle[1] = y_weighted_mean

        lower_row = int(best_particle[1] - self.template.shape[0] // 2)
        upper_row =  int(best_particle[1] + self.template.shape[0] // 2)

        lower_column = int(best_particle[0] - self.template.shape[1] // 2)
        upper_column = int(best_particle[0] + self.template.shape[1] // 2)

        # Grab the best_estimate frame
        best_estimate = self.frame[lower_row:upper_row, lower_column:upper_column]
        best_estimate = cv2.resize(best_estimate, (self.template.shape[1], self.template.shape[0]), interpolation=cv2.INTER_CUBIC)

        self.template = cv2.addWeighted(best_estimate, (self.alpha), self.template, (1 - self.alpha), 0)
        self.video_out.write(self.template)

        super().process(frame)

class CustomParticleFilter(AppearanceModelPF):
    """A variation of particle filter tracker that incorporates more dynamics."""

    def __init__(self, frame, template, **kwargs):
        super(CustomParticleFilter, self).__init__(frame, template, **kwargs)

        # Incorporate additional states as different motion model will be used
        x_lower_vel, x_upper_vel = kwargs.get('init_vel_x', (0, 5))
        y_lower_vel, y_upper_vel = kwargs.get('init_vel_y', (-1, 1))

        rand_vel_x = np.random.uniform(x_lower_vel, x_upper_vel, self.num_particles)
        rand_vel_y = np.random.uniform(y_lower_vel, y_upper_vel, self.num_particles)

        self.particles_velocity = np.vstack((rand_vel_x, rand_vel_y))
        self.particles_velocity = self.particles_velocity.T
        self.alpha = 1
        self.count = 0

    def process(self, frame):
        """Processes a video frame (image) and updates the filter's state.
        """
        
        best_particle = np.zeros(2)

        x_weighted_mean = 0
        y_weighted_mean = 0

        for i in range(self.num_particles):
           x_weighted_mean += self.particles[i, 0] * self.weights[i]
           y_weighted_mean += self.particles[i, 1] * self.weights[i]

        best_particle[0] = x_weighted_mean
        best_particle[1] = y_weighted_mean

        lower_row = int(best_particle[1] - self.template.shape[0] // 2)
        upper_row =  int(best_particle[1] + self.template.shape[0] // 2)

        lower_column = int(best_particle[0] - self.template.shape[1] // 2)
        upper_column = int(best_particle[0] + self.template.shape[1] // 2)

        # Grab the best_estimate frame
        best_estimate = self.frame[lower_row:upper_row, lower_column:upper_column]


        best_estimate = cv2.resize(best_estimate, (self.template.shape[1], self.template.shape[0]), interpolation=cv2.INTER_CUBIC)

        # if (self.count == 15):
        #     self.template = best_estimate
        #     self.count = 0
        # elif ((self.count != 0) and (not self.count % 5)):
        #     self.template = cv2.addWeighted(best_estimate, (self.alpha), self.template, (1 - self.alpha), 0)
        self.template = cv2.addWeighted(best_estimate, (self.alpha), self.template, (1 - self.alpha), 0)

        self.count += 1
        norm = 0
        
        for i in range(0, self.num_particles):
            try:
                # Update particles velocities
                self.particles_velocity[i, 0] = self.particles_velocity[i, 0] + np.abs(np.random.normal(0, 3, 1))
                self.particles_velocity[i, 1] = self.particles_velocity[i, 1] + np.random.normal(-1, 1, 1)

                # Update particles position
                self.particles[i, 0] = self.particles[i, 0] + self.particles_velocity[i, 0]
                self.particles[i, 1] = self.particles[i, 1] + self.particles_velocity[i, 1]

                # p(z | x)
                lower_row = int(self.particles[i, 1] - self.template.shape[0] // 2)
                upper_row = int(self.particles[i, 1] + self.template.shape[0] // 2)

                lower_column = int(self.particles[i, 0] - self.template.shape[1] // 2)
                upper_column = int(self.particles[i, 0] + self.template.shape[1] // 2)

                frame_cutout = self.frame[lower_row:upper_row, lower_column:upper_column]
                frame_cutout = cv2.resize(frame_cutout, (self.template.shape[1], self.template.shape[0]), interpolation=cv2.INTER_CUBIC)         
                
                particle_mse = self.get_error_metric(self.template, frame_cutout)
                self.weights[i] = np.exp(-particle_mse/(2*self.sigma_exp**2))

                # Update normalization factor
                norm += self.weights[i]

            except:
                self.weights[i] = 0
                continue

        # Normalize weifhts
        self.weights = self.weights / norm

        # Resample particles
        self.particles = self.resample_particles()
        self.frame = frame