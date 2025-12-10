import cv2
import numpy as np

# Global variables
hsv_value = None
rgb_value = None
mouse_x, mouse_y = 0, 0

def mouse_callback(event, x, y, flags, param):
    """Callback function for mouse events"""
    global hsv_value, rgb_value, mouse_x, mouse_y
    
    if event == cv2.EVENT_MOUSEMOVE or event == cv2.EVENT_LBUTTONDOWN:
        mouse_x, mouse_y = x, y
        
        # Get the pixel color at mouse position
        if 0 <= y < param['image'].shape[0] and 0 <= x < param['image'].shape[1]:
            # Get BGR values
            bgr_pixel = param['image'][y, x]
            rgb_value = bgr_pixel[::-1]  # Convert BGR to RGB
            
            # Get HSV values
            hsv_pixel = param['hsv_image'][y, x]
            hsv_value = hsv_pixel
            
            # Print when clicked
            if event == cv2.EVENT_LBUTTONDOWN:
                h, s, v = hsv_value
                r, g, b = rgb_value
                # print(f"\n--- Clicked at Position: ({mouse_x}, {mouse_y}) ---")
                print(f"HSV: H={h}, S={s}, V={v}")
                # print(f"RGB: R={r}, G={g}, B={b}")
                # print(f"Lower HSV: ({max(0, h-10)}, {max(0, s-40)}, {max(0, v-40)})")
                # print(f"Upper HSV: ({min(179, h+10)}, {min(255, s+40)}, {min(255, v+40)})")
                # print("-" * 50)

def main():
    # Load the image
    image_path = input("Enter the path to your image: ")
    image = cv2.imread(image_path)
    
    if image is None:
        print("Error: Could not load image. Please check the path.")
        return
    
    # Resize image if too large
    max_width = 1200
    max_height = 800
    height, width = image.shape[:2]
    
    if width > max_width or height > max_height:
        scale = min(max_width/width, max_height/height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        image = cv2.resize(image, (new_width, new_height))
    
    # Convert to HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Create window and set mouse callback
    window_name = 'HSV Color Detector - Hover to detect colors'
    cv2.namedWindow(window_name)
    
    param = {'image': image, 'hsv_image': hsv_image}
    cv2.setMouseCallback(window_name, mouse_callback, param)
    
    print("\n=== HSV Color Detector ===")
    print("Instructions:")
    print("- Move your mouse over the image to detect colors")
    print("- Click on the image to print HSV values to terminal")
    print("- Press 'q' or 'ESC' to quit")
    print("- Press 's' to save current HSV values to file")
    print("==========================\n")
    
    while True:
        # Create a copy of the image for display
        display_image = image.copy()
        
        # Draw crosshair at mouse position
        if 0 <= mouse_y < image.shape[0] and 0 <= mouse_x < image.shape[1]:
            cv2.line(display_image, (mouse_x - 10, mouse_y), (mouse_x + 10, mouse_y), (0, 255, 0), 1)
            cv2.line(display_image, (mouse_x, mouse_y - 10), (mouse_x, mouse_y + 10), (0, 255, 0), 1)
            cv2.circle(display_image, (mouse_x, mouse_y), 5, (0, 255, 0), 1)
        
        # Create info panel
        info_panel = np.zeros((200, 400, 3), dtype=np.uint8)
        
        if hsv_value is not None and rgb_value is not None:
            # Display color sample
            color_sample = np.full((80, 200, 3), rgb_value[::-1].tolist(), dtype=np.uint8)  # BGR for display
            info_panel[10:90, 10:210] = color_sample
            
            # Display text information
            h, s, v = hsv_value
            r, g, b = rgb_value
            
            cv2.putText(info_panel, f"HSV: ({h}, {s}, {v})", (220, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(info_panel, f"RGB: ({r}, {g}, {b})", (220, 55), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(info_panel, f"Pos: ({mouse_x}, {mouse_y})", (220, 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Display HSV ranges for color detection
            cv2.putText(info_panel, "Suggested HSV Range:", (10, 110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
            
            lower_h = max(0, h - 10)
            upper_h = min(179, h + 10)
            lower_s = max(0, s - 40)
            upper_s = min(255, s + 40)
            lower_v = max(0, v - 40)
            upper_v = min(255, v + 40)
            
            cv2.putText(info_panel, f"Lower: ({lower_h}, {lower_s}, {lower_v})", (10, 135), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 255, 100), 1)
            cv2.putText(info_panel, f"Upper: ({upper_h}, {upper_s}, {upper_v})", (10, 160), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 255, 100), 1)
            
            cv2.putText(info_panel, "Press 's' to save | Click to print", (10, 190), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        
        # Display the images
        cv2.imshow(window_name, display_image)
        cv2.imshow('Color Info', info_panel)
        
        # Key handling
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q') or key == 27:  # 'q' or ESC
            break
        elif key == ord('s') and hsv_value is not None:
            # Save HSV values to file
            with open('hsv_values.txt', 'a') as f:
                h, s, v = hsv_value
                r, g, b = rgb_value
                # f.write(f"Position: ({mouse_x}, {mouse_y})\n")
                f.write(f"HSV: ({h}, {s}, {v})\n")
                # f.write(f"RGB: ({r}, {g}, {b})\n")
                # f.write(f"Lower HSV: ({max(0, h-10)}, {max(0, s-40)}, {max(0, v-40)})\n")
                # f.write(f"Upper HSV: ({min(179, h+10)}, {min(255, s+40)}, {min(255, v+40)})\n")
                f.write("-" * 50 + "\n")
            print(f"Saved HSV values to hsv_values.txt")
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()