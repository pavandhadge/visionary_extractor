import cv2
import pytesseract
import re
import requests
import numpy as np
from io import BytesIO
from PIL import Image, ImageEnhance
import easyocr
import pandas as pd
import os
def download_image(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content))
        return image  # Return as PIL Image for further processing
    except requests.RequestException as e:
        print(f"Error downloading image from {url}: {e}")
        return None

def preprocess_image(image):
    try:
        # Convert the image to RGB (if it's not already)
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Enhance the image contrast
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(2)  # Increase contrast by a factor of 2

        # Apply sharpening to make the text clearer
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(2)  # Sharpen the image by a factor of 2

        # Convert PIL image to a NumPy array for additional OpenCV processing
        np_image = np.array(image)

        # Optional: Apply further denoising (useful for images with noise)
        denoised = cv2.fastNlMeansDenoisingColored(np_image, None, 10, 10, 7, 21)

        # Convert back to PIL Image format for Tesseract
        processed_image = Image.fromarray(denoised)
        return processed_image
    except Exception as e:
        print(f"Error in preprocess_image: {e}")
        return None

def extract_text(image):
    tesseract_text = pytesseract.image_to_string(image, config='--psm 6')
    reader = easyocr.Reader(['en'])
    easyocr_result = reader.readtext(np.array(image))  # Convert PIL image to numpy array
    easyocr_text = ' '.join([detection[1] for detection in easyocr_result])
    return f"{tesseract_text} {easyocr_text}"

def extract_entity_data(text, entity_name):
    patterns = {
        "width": r'(\d+(?:\.\d+)?)\s*(inch|in|inches|cm|centimeter|centimeters|centimetre|centimetres|mm|millimeter|millimeters|millimetre|millimetres)',
        "depth": r'(\d+(?:\.\d+)?)\s*(inch|in|inches|cm|centimeter|centimeters|centimetre|centimetres|mm|millimeter|millimeters|millimetre|millimetres)',
        "height": r'(\d+(?:\.\d+)?)\s*(inch|in|inches|cm|centimeter|centimeters|centimetre|centimetres|mm|millimeter|millimeters|millimetre|millimetres)',
        "item_volume": r'(\d+(?:\.\d+)?)\s*(cup|cups|ml|millilitre|millilitres|milliliter|milliliters|fluid ounce|fluid ounces)',
        "item_weight": r'(\d+(?:\.\d+)?)\s*(pound|lb|pounds|kg|kilogram|kilograms|g|gram|grams|mg|milligram|milligrams|oz|ounce|ounces)',
        "maximum_weight_recommendation": r'(\d+(?:\.\d+)?)\s*(pound|lb|pounds|kg|kilogram|kilograms|g|gram|grams|mg|milligram|milligrams|oz|ounce|ounces)',
        "voltage": r'(\d+(?:\.\d+)?)\s*(volt|v)',
        "wattage": r'(\d+(?:\.\d+)?)\s*(watt|w)'
    }

    pattern = patterns.get(entity_name.lower())
    if pattern:
        matches = re.findall(pattern, text, re.IGNORECASE)
        print(f"Matches found: {matches}")  # Debugging line
        if matches:
            match = matches[0]
            if len(match) >= 2:
                value, unit = match[:2]
                return float(value), unit

    return None, None

def normalize_unit(value, unit, entity_name):
    unit = unit.lower()
    if entity_name in ["width", "depth", "height"]:
        if unit in ['in', 'inch', 'inches']:
            return value, 'inch'
        elif unit in ['cm', 'centimeter', 'centimeters', 'centimetre', 'centimetres']:
            return value, 'centimetre'
        elif unit in ['mm', 'millimeter', 'millimeters', 'millimetre', 'millimetres']:
            return value, 'millimetre'
    elif entity_name == "item_volume":
        if unit in ['cup', 'cups']:
            return value, 'cup'
        elif unit in ['ml', 'millilitre', 'millilitres', 'milliliter', 'milliliters']:
            return value, 'millilitre'
        elif unit in ['fluid ounce', 'fluid ounces']:
            return value, 'fluid ounce'
    elif entity_name in ["item_weight", "maximum_weight_recommendation"]:
        if unit in ['lb', 'pound', 'pounds']:
            return value, 'pound'
        elif unit in ['kg', 'kilogram', 'kilograms']:
            return value, 'kilogram'
        elif unit in ['g', 'gram', 'grams']:
            return value, 'gram'
        elif unit in ['mg', 'milligram', 'milligrams']:
            return value, 'milligram'
        elif unit in ['oz', 'ounce', 'ounces']:
            return value, 'ounce'
    elif entity_name == "voltage":
        return value, 'volt'
    elif entity_name == "wattage":
        return value, 'watt'
    return value, unit

def process_image_url(url, entity_name):
    image = download_image(url)
    if image is None:
        return None

    preprocessed_image = preprocess_image(image)
    if preprocessed_image is None:
        return None

    extracted_text = extract_text(preprocessed_image)
    value, unit = extract_entity_data(extracted_text, entity_name)

    if value is not None and unit:
        normalized_value, normalized_unit = normalize_unit(value, unit, entity_name)
        return f"{normalized_value:.2f} {normalized_unit}"

    return 'Not found'


def modelmain(dataset):
    results = []
    for _, row in dataset.iterrows():
        print(f"Processing {row['image_link']} for {row['entity_name']}...")
        entity_value = process_image_url(row['image_link'], row['entity_name'])
        if entity_value == 'Not found':
            entity_value = ""


        results.append({
            'index': row['index'],
            # 'image_link': row['image_link'],
            # 'group_id': row['group_id'],
            # 'entity_name': row['entity_name'],
            'prediction': entity_value
        })

    results_df = pd.DataFrame(results)
    output_file = 'extraction_results.csv'
    
    # Check if the file already exists
    if os.path.exists(output_file):
        # Append to the existing file
        results_df.to_csv(output_file, mode='a', header=False, index=False)
    else:
        # Create a new file and write the header
        results_df.to_csv(output_file, index=False, header=['index', 'prediction'])
    
    print("\nExtraction complete. Results saved to 'extraction_results.csv'.")
    return results



def predictor(image_link, entity_name,index,group_id):
    '''
    Call your model/approach here
    '''
    
    # Create a temporary dataset with a single row
    temp_df = pd.DataFrame({
        'index': [index],
        'image_link': [image_link],
        'group_id': [group_id],  # Assuming you don't need it for a single prediction
        'entity_name': [entity_name]
    })

    # Process the temporary dataset
    results = modelmain(temp_df)
    
    # Get the result for the single image
    result = results[0]['prediction']
    return result

if __name__ == "__main__":
    DATASET_FOLDER = 'C:\\Users\\Dipak\\OneDrive - South Indian Education Society\\Desktop\\cognition_24\\cognition_24'
    
    test = pd.read_csv(os.path.join(DATASET_FOLDER, 'sample.csv'))
    
    test['prediction'] = test.apply(
        lambda row: predictor(row['image_link'], row['entity_name'],row['index'],row['group_id']), axis=1)
    
    output_filename = os.path.join(DATASET_FOLDER, 'test_out.csv')
    test[['index', 'prediction']].to_csv(output_filename, index=False)