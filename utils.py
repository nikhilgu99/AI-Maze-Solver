import cv2

def draw_path_and_save(image_path, path_coords, output_filename="solved_maze.bmp"):
    """
    Loads the original maze image, draws the solved path, and saves the result.
    """
    # Load image in color mode so the path stands out
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    
    if img is None:
        print(f"[!] Error: Could not load image for drawing: {image_path}")
        return False
        
    # Define the color for the path in BGR format (Red).
    path_color = (0, 0, 255)
    
    # Color every pixel in the solved path
    for row, col in path_coords:
        img[row, col] = path_color
        
    # Save the modified image
    cv2.imwrite(output_filename, img)
    return True