from PIL import Image, ImageDraw
import numpy as np

def create_grass_texture(size=512):
    """Create a realistic grass texture."""
    img = Image.new('RGB', (size, size), (34, 139, 34))  # Forest green base
    draw = ImageDraw.Draw(img)
    
    # Add grass blade patterns
    for _ in range(1000):
        x = np.random.randint(0, size)
        y = np.random.randint(0, size)
        length = np.random.randint(5, 15)
        angle = np.random.uniform(0, np.pi)
        color = (
            np.random.randint(20, 100),
            np.random.randint(100, 200),
            np.random.randint(20, 100)
        )
        
        end_x = x + length * np.cos(angle)
        end_y = y + length * np.sin(angle)
        draw.line([(x, y), (end_x, end_y)], fill=color, width=1)
    
    img.save('textures/grass.jpg')

def create_rock_texture(size=512):
    """Create a realistic rock texture."""
    img = Image.new('RGB', (size, size), (128, 128, 128))  # Gray base
    draw = ImageDraw.Draw(img)
    
    # Add rock patterns
    for _ in range(500):
        x = np.random.randint(0, size)
        y = np.random.randint(0, size)
        radius = np.random.randint(5, 20)
        color = np.random.randint(100, 160)
        draw.ellipse(
            [(x-radius, y-radius), (x+radius, y+radius)],
            fill=(color, color, color)
        )
    
    img.save('textures/rock.jpg')

def create_sky_texture(size=512):
    """Create a realistic sky texture."""
    img = Image.new('RGB', (size, size), (135, 206, 235))  # Sky blue base
    draw = ImageDraw.Draw(img)
    
    # Add cloud patterns
    for _ in range(50):
        x = np.random.randint(0, size)
        y = np.random.randint(0, size//3)  # Clouds mainly at the top
        radius = np.random.randint(20, 50)
        color = np.random.randint(240, 255)
        draw.ellipse(
            [(x-radius, y-radius), (x+radius, y+radius)],
            fill=(color, color, color),
            outline=(color, color, color)
        )
    
    img.save('textures/sky.jpg')

if __name__ == "__main__":
    print("Generating textures...")
    create_grass_texture()
    create_rock_texture()
    create_sky_texture()
    print("Textures generated successfully!") 