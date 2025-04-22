import numpy as np
import matplotlib.pyplot as plt 
import pygame

def circular_path(radius, num_points=100, track_width=2.0): 
    """
    Generate a circular path with a specified radius and width and returns the left and right cones and the start cones
    that specifies the track boundaries and car start_point.
    """
    theta = np.linspace(2*np.pi/num_points, 2 * np.pi, num_points)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)

    # Left and right cones
    left_cones = np.array([
        (radius - track_width / 2) * np.cos(theta),
        (radius - track_width / 2) * np.sin(theta)
    ]).T
    right_cones = np.array([
        (radius + track_width / 2) * np.cos(theta),
        (radius + track_width / 2) * np.sin(theta)
    ]).T
    start_cones = np.array([
        [(radius - track_width / 2) * np.cos(0), (radius - track_width / 2) * np.sin(0)],
        [(radius + track_width / 2) * np.cos(0), (radius + track_width / 2) * np.sin(0)]
    ]).T
    
    start_point = np.array([radius * np.cos(0), radius * np.sin(0), np.pi/2, 0.0, 0.0])

    return left_cones, right_cones, start_cones, start_point

def plot_track(left_cones, right_cones, start_cones):
    """
    Plot the track defined by the left and right cones.
    """
    plt.figure(figsize=(8, 8))
    plt.plot(left_cones[:, 0], left_cones[:, 1], 'b.', label='Left Cone')
    plt.plot(right_cones[:, 0], right_cones[:, 1], 'y.', label='Right Cone')
    plt.plot(start_cones[0], start_cones[1], 'o', color='orange', label='Start Cone')
    plt.axis('equal')
    plt.title('Circular Track')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.legend()
    plt.grid()
    plt.show()
    
def pygame_show(left_cones, right_cones, start_cones, start_position):
    pygame.init()
    screen_width = 1000
    screen_height = 1000
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Circular Track")
    clock = pygame.time.Clock()
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.fill((255, 255, 255))  # Fill the screen with white

        # Draw left cones
        for cone in left_cones:
            pygame.draw.circle(screen, (0, 0, 255), (int(cone[0] * 80 + screen_width / 2), int(-cone[1] * 80 + screen_height / 2)), 5)

        # Draw right cones
        for cone in right_cones:
            pygame.draw.circle(screen, (255, 255, 0), (int(cone[0] * 80 + screen_width / 2), int(-cone[1] * 80 + screen_height / 2)), 5)

        # Draw start cones
        for cone in start_cones.T:
            pygame.draw.circle(screen, (255, 165, 0), (int(cone[0] * 80 + screen_width / 2), int(-cone[1] * 80 + screen_height / 2)), 5)
            
        # Draw car's start position
        car_surface = pygame.Surface((40, 20), pygame.SRCALPHA)  # Create a transparent surface
        car_surface.fill((255, 0, 0))  # Fill the car surface with red (or any color)

        # Rotate the car surface
        rotated_car = pygame.transform.rotate(car_surface, -start_position[2] * 180 / np.pi)

        # Get the rotated car's rect and set its center
        rotated_car_rect = rotated_car.get_rect(center=(int(start_position[0] * 80 + screen_width / 2), 
                                                        int(-start_position[1] * 80 + screen_height / 2)))

        # Blit the rotated car onto the screen
        screen.blit(rotated_car, rotated_car_rect)
        

        pygame.display.flip()
        clock.tick(60)
    

if __name__ == "__main__":
    radius = 5.0
    num_points = 100
    track_width = 2.0

    left_cones, right_cones, start_cones, start_position = circular_path(radius, num_points, track_width)
    plot_track(left_cones, right_cones, start_cones)
    pygame_show(left_cones, right_cones, start_cones, start_position[:3])