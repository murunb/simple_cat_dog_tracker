import cv2
import filters
import os
import numpy as np
import videohelper

kenabletemplatesave = False
kenabletemplatesave2 = False

# I/O and Video Specifications
video_dir = "input_video"
video_file = "catanddog1.mp4"
video_file_2 = "catanddog2.mp4"

output_dir = "./"
fps = 20

my_video = os.path.join(video_dir, video_file)
my_video_2 = os.path.join(video_dir, video_file_2)

def run_particle_filter_1(filter_class, **kwargs):
    """Runs a particle filter on a given video and template.
    """
    try:
        print("Running Particle Filter")

        # Set initial frames
        dog_frame_num = 8
        cat_frame_num = 180

        # Prepare dog template
        dog_template = {'y': 525, 'x': 735, 'w': 91, 'h': 150}
        dog_frame_filepath = "dog_frame_1.jpeg"
        dog_frame = cv2.imread(dog_frame_filepath)

        # Prepare cat template
        cat_template = {'y': 529, 'x': 609, 'w': 100, 'h': 120}
        cat_frame_filepath = "cat_frame_1.jpeg"
        cat_frame = cv2.imread(cat_frame_filepath)

        # Extract all templates
        template_1 = dog_frame[dog_template['y']:
                dog_template['y'] + dog_template['h'],
                dog_template['x']:
                dog_template['x'] + dog_template['w']]
        
        template_2 = cat_frame[cat_template['y']:
            cat_template['y'] + cat_template['h'],
            cat_template['x']:
            cat_template['x'] + cat_template['w']]
        
        image_gen = videohelper.video_frame_generator(my_video)
        current_image = image_gen.__next__()
        h, w, _ = current_image.shape

        out_path = "PF-catanddog_1.mp4"

        video_out = videohelper.mp4_video_writer(out_path, (w, h), fps)

        frame_number = 0
        kwargs["template_coords"] = dog_template
        kwargs["video_mode"] = "PF-dog_template_1.mp4"
        pf_dog = filter_class(current_image, template_1, **kwargs)

        kwargs["template_coords"] = cat_template
        kwargs["video_mode"] = "PF-cat_template_1.mp4"
        pf_cat = filters.AppearanceModelPF(current_image, template_2, **kwargs)


        while current_image is not None:
            print("Processing frame {}.".format(frame_number))

            # Increment frame
            frame_number += 1
            current_image = image_gen.__next__()
            out_frame = current_image.copy()

            if (frame_number <= 62):
                    
                # Process frame
                pf_dog.process(current_image)

                out_frame = current_image.copy()
                pf_dog.render(out_frame)

            if (frame_number > 170):
                pf_cat.process(current_image)
                out_frame = current_image.copy()
                pf_cat.render(out_frame)

            video_out.write(out_frame)

        video_out.release()
    except:
        pass

def run_particle_filter_2(filter_class, **kwargs):
    """Runs a particle filter on a given video and template.
    """
    try:
        print("Running Particle Filter")

        # Set initial frames
        dog_frame_num = 11
        cat_frame_num = 68

        # Prepare dog template
        dog_template = {'y': 615, 'x': 60, 'w': 100, 'h': 140}
        dog_frame_filepath = "dog_frame_2.jpeg"
        dog_frame = cv2.imread(dog_frame_filepath)

        # Prepare cat template
        cat_template = {'y': 570, 'x': 0, 'w': 105, 'h': 140}
        cat_frame_filepath = "cat_frame_2.jpeg"
        cat_frame = cv2.imread(cat_frame_filepath)


        # Extract all templates
        template_1 = dog_frame[dog_template['y']:
                dog_template['y'] + dog_template['h'],
                dog_template['x']:
                dog_template['x'] + dog_template['w']]

        template_2 = cat_frame[cat_template['y']:
            cat_template['y'] + cat_template['h'],
            cat_template['x']:
            cat_template['x'] + cat_template['w']]

        image_gen = videohelper.video_frame_generator(my_video_2)
        current_image = image_gen.__next__()
        h, w, _ = current_image.shape

        out_path = "PF-catanddog_2.mp4"
        video_out = videohelper.mp4_video_writer(out_path, (w, h), fps)

        frame_number = 0
        kwargs["template_coords"] = dog_template
        kwargs["video_mode"] = "PF-dog_template_2.mp4"
        pf_dog = filter_class(current_image, template_1, **kwargs)

        kwargs["template_coords"] = cat_template
        kwargs["video_mode"] = "PF-cat_template_2.mp4"

        pf_cat = filters.CustomParticleFilter(current_image, template_2, **kwargs)

        while current_image is not None:
            print("Processing frame {}.".format(frame_number))

            # Increment frame
            frame_number += 1
            current_image = image_gen.__next__()
            out_frame = current_image.copy()

            if (frame_number > 10 and frame_number <= 60):
                    
                # Process frame
                pf_dog.process(current_image)

                out_frame = current_image.copy()
                pf_dog.render(out_frame)

            if (frame_number > 68):
                pf_cat.process(current_image)
                out_frame = current_image.copy()
                pf_cat.render(out_frame)

            video_out.write(out_frame)

        video_out.release()
    except:
        pass



def run_KF_1():
    print("Running Kalman Filter")

    # Set initial frames
    dog_frame_num = 8
    cat_frame_num = 180

    # Prepare dog template
    dog_template = {'y': 525, 'x': 735, 'w': 91, 'h': 150}
    dog_frame_filepath = "dog_frame_1.jpeg"
    dog_frame = cv2.imread(dog_frame_filepath)

    # Prepare cat template
    cat_template = {'y': 529, 'x': 609, 'w': 100, 'h': 120}
    cat_frame_filepath = "cat_frame_1.jpeg"
    cat_frame = cv2.imread(cat_frame_filepath)

    # Extract all templates
    template_1 = dog_frame[dog_template['y']:
            dog_template['y'] + dog_template['h'],
            dog_template['x']:
            dog_template['x'] + dog_template['w']]
    
    cv2.imshow("Dog template", template_1)
    cv2.waitKey(0)

    template_2 = cat_frame[cat_template['y']:
        cat_template['y'] + cat_template['h'],
        cat_template['x']:
        cat_template['x'] + cat_template['w']]
    cv2.imshow("Cat template", template_2)
    cv2.waitKey(0)

    # Initialize position
    init_pos = {'x': 735, 'y': 650}
    noise = {'x': 2.5, 'y': 2.5}

    # Process and Measurement Noise
    Q = 0.3 * np.eye(4)
    R = 0.1 * np.eye(2)

    kf_dog = filters.KalmanFilter(init_pos['x'], init_pos['y'], Q, R)
    kf_cat = filters.KalmanFilter(init_pos['x'], init_pos['y'], Q, R)
    
    image_gen = videohelper.video_frame_generator(my_video)
    current_image = image_gen.__next__()
    h, w, _ = current_image.shape

    out_path = "KF-catanddog_1.mp4"
    video_out = videohelper.mp4_video_writer(out_path, (w, h), fps)

    frame_number = 0

    while current_image is not None:
        try:
            print("Processing frame {}.".format(frame_number))

            if kenabletemplatesave:
                # Extract template
                if (frame_number == dog_frame_num):
                    cv2.imwrite("dog_frame_1.jpeg", current_image)

                if (frame_number == cat_frame_num):
                    cv2.imwrite("cat_frame_1.jpeg", current_image)


            # Increment frame
            frame_number += 1
            current_image = image_gen.__next__()
            out_frame = current_image.copy()

            if (frame_number <= 62):
                corr_map = cv2.matchTemplate(current_image, template_1, cv2.TM_SQDIFF)
                z_y, z_x = np.unravel_index(np.argmin(corr_map), corr_map.shape)
                z_w = dog_template['w']
                z_h = dog_template['h']
                z_x += z_w // 2 + np.random.normal(0, noise['x'])
                z_y += z_h // 2 + np.random.normal(0, noise['y'])
                x, y = kf_dog.process(z_x, z_y)
                cv2.rectangle(out_frame, (int(x) - z_w // 2, int(y) - z_h // 2),
                    (int(x) + z_w // 2, int(y) + z_h // 2),
                    (0, 0, 255), 2)

            if (frame_number > 170):
                corr_map_2 = cv2.matchTemplate(current_image, template_2, cv2.TM_SQDIFF)
                z_y_2, z_x_2 = np.unravel_index(np.argmin(corr_map_2), corr_map_2.shape)
                z_w_2 = cat_template['w']
                z_h_2 = cat_template['h']
                z_x_2 += z_w_2 // 2 + np.random.normal(0, noise['x'])
                z_y_2 += z_h_2 // 2 + np.random.normal(0, noise['y'])

                x_2, y_2 = kf_cat.process(z_x_2, z_y_2)

                cv2.rectangle(out_frame, (int(x_2) - z_w_2 // 2, int(y_2) - z_h_2 // 2),
                    (int(x_2) + z_w_2 // 2, int(y_2) + z_h_2 // 2),
                    (255, 0, 0), 2)

            video_out.write(out_frame)
        except:
            break
    video_out.release()

def run_KF_2():
    try:
        print("Running Kalman Filter for second video")

        # Set initial frames
        dog_frame_num = 11
        cat_frame_num = 68

        k_counter = 0


        # Prepare dog template
        dog_template = {'y': 615, 'x': 60, 'w': 100, 'h': 140}
        dog_frame_filepath = "dog_frame_2.jpeg"
        dog_frame = cv2.imread(dog_frame_filepath)

        # Prepare cat template
        cat_template = {'y': 600, 'x': 2, 'w': 95, 'h': 100}
        cat_frame_filepath = "cat_frame_2.jpeg"
        cat_frame = cv2.imread(cat_frame_filepath)


        # Extract all templates
        template_1 = dog_frame[dog_template['y']:
                dog_template['y'] + dog_template['h'],
                dog_template['x']:
                dog_template['x'] + dog_template['w']]
        
        #cv2.imshow("Dog template", template_1)
        #cv2.waitKey(0)

        template_2 = cat_frame[cat_template['y']:
            cat_template['y'] + cat_template['h'],
            cat_template['x']:
            cat_template['x'] + cat_template['w']]
        
        #cv2.imshow("Cat template", template_2)
        #cv2.waitKey(0)

        # Initialize position
        init_pos = {'x': 50, 'y': 550}
        noise = {'x': 1, 'y': 1}

        # Process and Measurement Noise
        Q = 0.3 * np.eye(4)
        R = 0.1 * np.eye(2)

        kf_dog = filters.KalmanFilter(init_pos['x'], init_pos['y'], Q, R)
        kf_cat = filters.KalmanFilter(init_pos['x'], init_pos['y'], Q, R)
        
        image_gen = videohelper.video_frame_generator(my_video_2)
        current_image = image_gen.__next__()
        h, w, _ = current_image.shape
        h_template, w_template, _ = template_1.shape
        h_template_2, w_template_2, _ = template_2.shape

        out_path = "KF-catanddog_2.mp4"
        out_path_template_dog = "KF-template_update_dog.mp4"
        out_path_template_cat = "KF-template_update_cat.mp4"
        video_out = videohelper.mp4_video_writer(out_path, (w, h), fps)
        video_template_dog = videohelper.mp4_video_writer(out_path_template_dog, (w_template, h_template), fps)
        video_template_cat = videohelper.mp4_video_writer(out_path_template_cat, (w_template_2, h_template_2), fps)

        frame_number = 0

        while current_image is not None:
            print("Processing frame {}.".format(frame_number))

            if kenabletemplatesave:
                # Extract template
                if (frame_number == dog_frame_num):
                    cv2.imwrite("dog_frame_2.jpeg", current_image)

                if (frame_number == cat_frame_num):
                    cv2.imwrite("cat_frame_2.jpeg", current_image)


            # Increment frame
            frame_number += 1
            current_image = image_gen.__next__()
            out_frame = current_image.copy()

            alpha = 1
            if (frame_number > 10 and frame_number <= 60):

                corr_map = cv2.matchTemplate(current_image, template_1, cv2.TM_SQDIFF)
                z_y, z_x = np.unravel_index(np.argmin(corr_map), corr_map.shape)
                z_w = dog_template['w']
                z_h = dog_template['h']
                z_x += z_w // 2 + np.random.normal(0, noise['x'])
                z_y += z_h // 2 + np.random.normal(0, noise['y'])
                x, y = kf_dog.process(z_x, z_y)

                lower_row = int(y - template_1.shape[0] // 2)
                upper_row = int(y + template_1.shape[0] // 2)

                lower_column = int(x - template_1.shape[1] // 2)
                upper_column = int(x + template_1.shape[1] // 2)

                # Correct dimensions
                while (lower_row < 0):
                    lower_row += 1
                    upper_row += 1
                
                while (upper_row > current_image.shape[0]):
                    lower_row -= 1
                    upper_row -= 1
                
                while (lower_column < 0):
                    lower_column += 1
                    upper_column += 1
                
                while (upper_column > current_image.shape[1]):
                    lower_column -= 1
                    upper_column -= 1

                # Grab the best_estimate frame
                best_estimate = current_image[lower_row:upper_row, lower_column:upper_column]

                best_estimate = cv2.resize(best_estimate, (template_1.shape[1], template_1.shape[0]), interpolation=cv2.INTER_CUBIC)

                if ((k_counter != 0 ) and (not k_counter % 5)):
                    template_1 = cv2.addWeighted(best_estimate, alpha, template_1, (1 - alpha), 0)
                    k_counter = 0

                k_counter += 1
                video_template_dog.write(template_1)

                cv2.rectangle(out_frame, (int(x) - z_w // 2, int(y) - z_h // 2),
                    (int(x) + z_w // 2, int(y) + z_h // 2),
                    (0, 0, 255), 2)

            if (frame_number > 68):
                corr_map_2 = cv2.matchTemplate(current_image, template_2, cv2.TM_SQDIFF)
                z_y_2, z_x_2 = np.unravel_index(np.argmin(corr_map_2), corr_map_2.shape)
                z_w_2 = cat_template['w']
                z_h_2 = cat_template['h']
                z_x_2 += z_w_2 // 2 + np.random.normal(0, noise['x'])
                z_y_2 += z_h_2 // 2 + np.random.normal(0, noise['y'])

                x_2, y_2 = kf_cat.process(z_x_2, z_y_2)

                lower_row_2 = int(y - template_2.shape[0] // 2)
                upper_row_2 = int(y + template_2.shape[0] // 2)

                lower_column_2 = int(x - template_2.shape[1] // 2)
                upper_column_2 = int(x + template_2.shape[1] // 2)

                # Correct dimensions
                while (lower_row_2 < 0):
                    lower_row_2 += 1
                    upper_row_2 += 1
                
                while (upper_row_2 > current_image.shape[0]):
                    lower_row_2 -= 1
                    upper_row_2 -= 1
                
                while (lower_column_2 < 0):
                    lower_column_2 += 1
                    upper_column_2 += 1
                
                while (upper_column_2 > current_image.shape[1]):
                    lower_column_2 -= 1
                    upper_column_2 -= 1


                # Grab the best_estimate frame
                best_estimate_2 = current_image[lower_row_2:upper_row_2, lower_column_2:upper_column_2]

                best_estimate_2 = cv2.resize(best_estimate_2, (template_2.shape[1], template_2.shape[0]), interpolation=cv2.INTER_CUBIC)

                template_2 = cv2.addWeighted(best_estimate_2, alpha, template_2, (1 - alpha), 0)
                video_template_cat.write(template_2)

                cv2.rectangle(out_frame, (int(x_2) - z_w_2 // 2, int(y_2) - z_h_2 // 2),
                    (int(x_2) + z_w_2 // 2, int(y_2) + z_h_2 // 2),
                    (255, 0, 0), 2)

            video_out.write(out_frame)
        video_out.release()
        video_template_dog.release()
        video_template_cat.release()
    except:
        pass

def run_PF_third():
    print("Running Particle Filter")

    # Number of particles
    num_particles = 500
    # Define the value of sigma for the measurement exponential equation 
    sigma_mse = 2 
    # Define the value of sigma for the particles movement (dynamics) 
    sigma_dyn = 30
    alpha = 0.1
    template_rect = []

    run_particle_filter_1(filters.AppearanceModelPF,
                        num_particles=num_particles, sigma_exp=sigma_mse,
                        sigma_dyn=sigma_dyn, alpha=alpha,
                        template_coords=template_rect, mse_method='notgray')

def run_PF_fourth():
    print("Running Particle Filter")

    # Number of particles
    num_particles = 1000
    # Define the value of sigma for the measurement exponential equation 
    sigma_mse = 3
    # Define the value of sigma for the particles movement (dynamics) 
    sigma_dyn = 40
    alpha = 0.01
    template_rect = []

    run_particle_filter_2(filters.ParticleFilter,
                        num_particles=num_particles, sigma_exp=sigma_mse,
                        sigma_dyn=sigma_dyn, alpha=alpha,
                        template_coords=template_rect, mse_method='notgray')


if __name__ == '__main__':
    #run_KF_1()
    #run_KF_2()
    #run_PF_third()
    run_PF_fourth()
