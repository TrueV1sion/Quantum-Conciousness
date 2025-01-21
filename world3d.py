import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
from scipy.ndimage import gaussian_filter

class BasicWorld3D:
    def __init__(self):
        pygame.init()
        display = (1920, 1080)
        pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
        pygame.display.set_caption("Realistic Terrain")
        
        # Enable depth testing and nice rendering
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_CULL_FACE)
        glCullFace(GL_BACK)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glShadeModel(GL_SMOOTH)
        glClearColor(0.5, 0.7, 1.0, 1.0)  # Sky blue background
        
        # Set up the camera
        gluPerspective(45, (display[0]/display[1]), 0.1, 500.0)
        glTranslatef(0.0, -5.0, -20.0)
        
        # Basic settings
        self.position = [0.0, 10.0, 0.0]
        self.rotation = [30.0, 0.0]  # Start with a view from above
        self.move_speed = 0.5
        
        # Generate terrain
        self.generate_terrain()
        
    def generate_terrain(self):
        """Generate realistic terrain using multiple noise layers."""
        size = 100  # Reduced size for better performance
        self.terrain_size = size
        heightmap = np.zeros((size, size))
        
        # Multiple octaves of noise for realistic terrain
        octaves = [
            {"freq": 1, "amp": 10.0},    # Base mountains
            {"freq": 2, "amp": 5.0},     # Hills
            {"freq": 4, "amp": 2.5},     # Large rocks
            {"freq": 8, "amp": 1.25},    # Small rocks
            {"freq": 16, "amp": 0.6}     # Surface detail
        ]
        
        for octave in octaves:
            freq = octave["freq"]
            amp = octave["amp"]
            noise = np.random.uniform(-1, 1, (size, size))
            noise = gaussian_filter(noise, sigma=1.0/freq)
            heightmap += noise * amp
        
        # Normalize heightmap
        heightmap = (heightmap - heightmap.min()) / (heightmap.max() - heightmap.min())
        
        # Create vertices and colors
        vertices = []
        colors = []
        
        for z in range(size - 1):
            for x in range(size - 1):
                # Get heights for this quad
                h00 = heightmap[z, x]
                h10 = heightmap[z, x+1]
                h01 = heightmap[z+1, x]
                h11 = heightmap[z+1, x+1]
                
                # Scale heights
                h00 *= 15.0  # Increased height scale
                h10 *= 15.0
                h01 *= 15.0
                h11 *= 15.0
                
                # Calculate vertices for two triangles
                v1 = [x-size/2, h00, z-size/2]
                v2 = [x+1-size/2, h10, z-size/2]
                v3 = [x-size/2, h01, z+1-size/2]
                v4 = [x+1-size/2, h11, z+1-size/2]
                
                # Add vertices
                vertices.extend([v1, v2, v3, v2, v4, v3])
                
                # Color based on height and slope
                for h in [h00, h10, h01, h10, h11, h01]:
                    if h < 2:  # Water
                        colors.append([0.0, 0.2, 0.8])
                    elif h < 3:  # Beach
                        colors.append([0.8, 0.8, 0.6])
                    elif h < 7:  # Grass
                        green = 0.4 + np.random.uniform(0, 0.2)  # Varied grass color
                        colors.append([0.2, green, 0.1])
                    elif h < 10:  # Rock
                        gray = 0.4 + np.random.uniform(0, 0.2)  # Varied rock color
                        colors.append([gray, gray, gray])
                    else:  # Snow
                        white = 0.9 + np.random.uniform(0, 0.1)  # Slightly varied snow
                        colors.append([white, white, white])
        
        self.terrain_vertices = vertices
        self.terrain_colors = colors
        
    def handle_input(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    return False
        
        keys = pygame.key.get_pressed()
        
        # Movement
        if keys[pygame.K_w]:
            glTranslatef(0, 0, 0.5)
        if keys[pygame.K_s]:
            glTranslatef(0, 0, -0.5)
        if keys[pygame.K_a]:
            glTranslatef(0.5, 0, 0)
        if keys[pygame.K_d]:
            glTranslatef(-0.5, 0, 0)
        if keys[pygame.K_SPACE]:
            glTranslatef(0, -0.5, 0)
        if keys[pygame.K_LSHIFT]:
            glTranslatef(0, 0.5, 0)
        
        # Rotation
        if keys[pygame.K_LEFT]:
            glRotatef(1, 0, 1, 0)
        if keys[pygame.K_RIGHT]:
            glRotatef(-1, 0, 1, 0)
        if keys[pygame.K_UP]:
            glRotatef(1, 1, 0, 0)
        if keys[pygame.K_DOWN]:
            glRotatef(-1, 1, 0, 0)
        
        return True
    
    def render(self):
        # Clear buffers
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        # Draw terrain
        glBegin(GL_TRIANGLES)
        for i in range(len(self.terrain_vertices)):
            glColor3fv(self.terrain_colors[i])
            glVertex3fv(self.terrain_vertices[i])
        glEnd()
        
        pygame.display.flip()
    
    def run(self):
        running = True
        clock = pygame.time.Clock()
        
        while running:
            running = self.handle_input()
            self.render()
            clock.tick(60)

if __name__ == "__main__":
    world = BasicWorld3D()
    world.run() 