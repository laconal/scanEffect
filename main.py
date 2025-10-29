"""
PDF Scan Effect Script - Consistent Effects Version
Converts a PDF to look like a scanned document with same effects on all pages

Requirements:
pip install pdf2image Pillow numpy img2pdf

Note: pdf2image requires poppler-utils
- Windows: Download from https://github.com/oschwartz10612/poppler-windows/releases and add bin folder to PATH
- Linux: sudo apt-get install poppler-utils
- Mac: brew install poppler
"""

from pdf2image import convert_from_path
from PIL import Image, ImageFilter, ImageEnhance, ImageDraw
import numpy as np
import img2pdf
import random
import os
import tempfile

def add_paper_texture(img_array, intensity=0.3, seed=None):
    """Add subtle paper texture"""
    if seed is not None:
        np.random.seed(seed)
    texture = np.random.normal(0, intensity * 5, img_array.shape[:2])
    texture = np.stack([texture] * 3, axis=2)
    return np.clip(img_array + texture, 0, 255).astype(np.uint8)

def add_gaussian_noise(img_array, sigma=10, seed=None):
    """Add Gaussian noise to simulate scan grain"""
    if seed is not None:
        np.random.seed(seed)
    noise = np.random.normal(0, sigma, img_array.shape)
    return np.clip(img_array + noise, 0, 255).astype(np.uint8)

def add_shadow_gradient(img, intensity=0.15):
    """Add subtle shadow gradient (common in scans)"""
    width, height = img.size
    gradient = Image.new('L', (width, height), 255)
    draw = ImageDraw.Draw(gradient)
    
    # Create gradient from corner
    gradient_size = min(width, height) // 4
    if gradient_size > 0:
        for i in range(gradient_size):
            alpha = int(255 - (i * intensity * 255 / gradient_size))
            alpha = max(0, min(255, alpha))  # Ensure valid range
            draw.rectangle([i, i, width-i-1, height-i-1], outline=alpha)
    
    gradient = gradient.filter(ImageFilter.GaussianBlur(radius=50))
    
    # Apply gradient
    img_array = np.array(img).astype(np.float32)
    gradient_array = np.array(gradient).astype(np.float32) / 255.0
    gradient_array = np.stack([gradient_array] * 3, axis=2)
    
    result = img_array * gradient_array
    return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))

def apply_scan_effect(img, scan_params, page_num=0):
    """
    Apply scan effects to a PIL Image using pre-determined parameters
    
    Args:
        img: PIL Image object
        scan_params: Dictionary with pre-determined effect parameters
        page_num: Page number (for consistent seeding per page)
    """
    # Convert to RGB if needed
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Store original size
    original_size = img.size
    
    # 1. Slight rotation (same for all pages)
    angle = scan_params['rotation_angle']
    img = img.rotate(angle, expand=True, fillcolor=(250, 250, 250), resample=Image.BICUBIC)
    
    # Crop back to original aspect ratio to avoid size inconsistencies
    if img.size != original_size:
        # Calculate crop box to center the rotated image
        left = (img.size[0] - original_size[0]) // 2
        top = (img.size[1] - original_size[1]) // 2
        right = left + original_size[0]
        bottom = top + original_size[1]
        
        # Ensure we don't exceed bounds
        left = max(0, left)
        top = max(0, top)
        right = min(img.size[0], right)
        bottom = min(img.size[1], bottom)
        
        img = img.crop((left, top, right, bottom))
        
        # Resize back to exact original size if needed
        if img.size != original_size:
            img = img.resize(original_size, Image.LANCZOS)
    
    # 2. Add paper texture (with page-specific seed for variation)
    if scan_params['add_texture']:
        img_array = np.array(img)
        img_array = add_paper_texture(img_array, seed=scan_params['base_seed'] + page_num)
        img = Image.fromarray(img_array)
    
    # 3. Add Gaussian noise (with page-specific seed for variation)
    img_array = np.array(img)
    img_array = add_gaussian_noise(img_array, sigma=scan_params['noise_sigma'], 
                                   seed=scan_params['base_seed'] + page_num + 1000)
    img = Image.fromarray(img_array)
    
    # 4. Add shadow gradient (same intensity for all)
    # if scan_params['add_shadows']:
    #     img = add_shadow_gradient(img, intensity=scan_params['shadow_intensity'])
    
    # 5. Adjust brightness (same for all)
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(scan_params['brightness_factor'])
    
    # 6. Adjust contrast (same for all)
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(scan_params['contrast_factor'])
    
    # 7. Add slight blur (same for all)
    img = img.filter(ImageFilter.GaussianBlur(radius=scan_params['blur_radius']))
    
    # 8. Slight sharpness reduction (same for all)
    enhancer = ImageEnhance.Sharpness(img)
    img = enhancer.enhance(scan_params['sharpness_factor'])
    
    return img

def generate_scan_parameters(options=None):
    """
    Generate consistent scan parameters for all pages
    
    Args:
        options: Dictionary with effect parameter ranges
    
    Returns:
        Dictionary with specific values to use for all pages
    """
    if options is None:
        options = {}
    
    """
    optimal:
    Scan parameters (consistent for all pages):
        Rotation angle: 0.81°
        Noise sigma: 8
        Blur radius: 0.5
        Brightness: 1.014
        Contrast: 1.382
        JPEG quality: 97
        Shadow intensity: 0.083
    """

    # Default ranges
    rotation_range = options.get('rotation_range', (-1, 1))
    noise_sigma = options.get('noise_sigma', 8)
    blur_radius = options.get('blur_radius', 0.5)
    brightness_range = options.get('brightness_range', (1.014, 1.014))
    contrast_range = options.get('contrast_range', (1.382, 1.382))
    jpeg_quality = options.get('jpeg_quality', (90, 100))
    add_shadows = options.get('add_shadows', True)
    add_texture = options.get('add_texture', True)
    base_seed = options.get('seed', random.randint(0, 10000))
    
    # Generate specific values (same for all pages)
    scan_params = {
        'rotation_angle': random.uniform(*rotation_range),
        'noise_sigma': noise_sigma,
        'blur_radius': blur_radius,
        'brightness_factor': random.uniform(*brightness_range),
        'contrast_factor': random.uniform(*contrast_range),
        'jpeg_quality': random.randint(*jpeg_quality),
        'shadow_intensity': random.uniform(0.05, 0.15),
        'sharpness_factor': 0.9,
        'add_shadows': add_shadows,
        'add_texture': add_texture,
        'base_seed': base_seed
    }
    
    return scan_params

def pdf_to_scanned_pdf(input_pdf, output_pdf, dpi=200, options=None):
    """
    Convert a PDF to look like a scanned document with consistent effects
    
    Args:
        input_pdf: Path to input PDF
        output_pdf: Path to output PDF
        dpi: Resolution for conversion (200-300 typical for scans)
        options: Dictionary with scan effect parameters
    """
    print(f"Converting PDF: {input_pdf}")
    print(f"DPI: {dpi}")
    
    # Generate consistent scan parameters for all pages
    scan_params = generate_scan_parameters(options)
    print(f"\nScan parameters (consistent for all pages):")
    print(f"  Rotation angle: {scan_params['rotation_angle']:.2f}°")
    print(f"  Noise sigma: {scan_params['noise_sigma']}")
    print(f"  Blur radius: {scan_params['blur_radius']}")
    print(f"  Brightness: {scan_params['brightness_factor']:.3f}")
    print(f"  Contrast: {scan_params['contrast_factor']:.3f}")
    print(f"  JPEG quality: {scan_params['jpeg_quality']}")
    print(f"  Shadow intensity: {scan_params['shadow_intensity']:.3f}\n")
    
    # Convert PDF to images
    print("Converting PDF pages to images...")
    pages = convert_from_path(input_pdf, dpi=dpi)
    print(f"Found {len(pages)} pages")
    
    # Create temporary directory for processed images
    temp_dir = tempfile.mkdtemp()
    temp_images = []
    
    try:
        # Process each page with same parameters
        for i, page in enumerate(pages, 1):
            print(f"Processing page {i}/{len(pages)}...")
            
            # Apply scan effects with consistent parameters
            scanned_page = apply_scan_effect(page, scan_params, page_num=i)
            
            # Save to temporary file with same JPEG quality
            temp_path = os.path.join(temp_dir, f"page_{i:04d}.jpg")
            scanned_page.save(temp_path, 'JPEG', 
                            quality=scan_params['jpeg_quality'], 
                            optimize=True)
            temp_images.append(temp_path)
        
        # Convert images back to PDF
        print("\nCreating output PDF...")
        with open(output_pdf, "wb") as f:
            f.write(img2pdf.convert(temp_images))
        
        print(f"✓ Scanned PDF created: {output_pdf}")
        print(f"  File size: {os.path.getsize(output_pdf) / 1024 / 1024:.2f} MB")
        
    finally:
        # Cleanup temporary files
        for temp_file in temp_images:
            try:
                os.remove(temp_file)
            except:
                pass
        try:
            os.rmdir(temp_dir)
        except:
            pass

# Example usage
if __name__ == "__main__":
    # Basic usage with default settings (same effects on all pages)
    pdf_to_scanned_pdf("Поручение 2.pdf", "output_scanned.pdf")
    
    # Custom settings for reproducible results (use same seed)
    # custom_options = {
    #     'rotation_range': (-1.5, 1.5),
    #     'noise_sigma': 5,
    #     'blur_radius': 0.3,
    #     'brightness_range': (0.95, 1.05),
    #     'contrast_range': (0.98, 1.02),
    #     'jpeg_quality': (85, 95),
    #     'add_shadows': True,
    #     'add_texture': True,
    #     'seed': 12345  # Use specific seed for reproducible results
    # }
    # pdf_to_scanned_pdf("input.pdf", "output_custom.pdf", dpi=200, options=custom_options)

