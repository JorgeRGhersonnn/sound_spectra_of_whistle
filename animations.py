from manim import * 
import numpy as np 

class SoundWave(ThreeDScene):
    def construct(self): 
        # Camera orientation 
        # Azimuth: -45 deg. Elevation: 30 deg.        
        elevation = 30*DEGREES
        phi = 90*DEGREES - elevation 
        theta = -45*DEGREES

        self.set_camera_orientation(phi=phi, theta=theta) # sets camera orientation to specified angles.

        # Create the pipe (cylindrical tube) over which sound waves travel. 
        pipe_length = 6 
        pipe_radius = 1 
        pipe = Cylinder( 
            radius=pipe_radius,
            height=pipe_length,
            direction=RIGHT, 
            fill_opacity=0.5,
            color= BLUE_A,
            resolution=(24, 24)
        )

        # add labels for angles 
        self.add_fixed_in_frame_mobjects(
            Text(f"Elevation: {elevation/DEGREES:.0f}°").to_corner(UL)
        )

        # create ''air particles'' 
        air_particles = VGroup()
        num_x = 30                  # particles along x-axis
        num_ring = 6               # particles per radial ring
        num_radii = 3               # number of concentric rings

        # wave parameters 
        amplitude = 0.3
        wave_number = 2.0 
        angular_frequency = 5.0 

        for i in range(num_x):
            x_base = interpolate(-pipe_length/2, pipe_length/2, i/num_x)
            for r_idx in range(1, num_radii + 1):
                r = (r_idx / num_radii) * (pipe_radius - 0.1)
                for j in range(num_ring * r_idx):
                    angle = (j / (num_ring * r_idx)) * TAU
                    y = r * np.cos(angle)
                    z = r * np.sin(angle)
                    
                    dot = Dot3D(
                        point=[x_base, y, z], 
                        radius=0.04, 
                        color=WHITE
                    )
                    # Store initial x for the wave calculation
                    dot.initial_x = x_base
                    air_particles.add(dot)

        def update_particles(p, dt):
            time = self.renderer.time
            for dot in p:
                # Longitudinal displacement: x = x0 + A * sin(kx - wt)
                displacement = amplitude * np.sin(
                    wave_number * dot.initial_x - angular_frequency * time
                )
                dot.set_x(dot.initial_x + displacement)
                
                # Visual cue: Change color based on compression
                # Closer to center of compression = Brighter/Redder
                if displacement > 0.2:
                    dot.set_color(RED_A)
                elif displacement < -0.2:
                    dot.set_color(BLUE_A)
                else:
                    dot.set_color(WHITE)

        air_particles.add_updater(update_particles)

        # Execution
        self.add(pipe, air_particles)
        self.wait(10)