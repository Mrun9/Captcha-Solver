from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import time, os, shutil
from PIL import Image
from io import BytesIO
from urllib.parse import urlparse

# === CONFIGURATION === #
HTML_FILE = "http://127.0.0.1:5000"
IMAGE_FOLDER = "captcha_images"
CAPTCHA_CENTER_WIDTH = 300
CAPTCHA_CENTER_HEIGHT = 300

import re
from selenium.webdriver.common.by import By

def extract_debug_info(driver):
    # Find the debug-info <div>
    debug_div = driver.find_element(By.CLASS_NAME, "debug-info")

    # Get all text and the <code> tag separately
    full_text = debug_div.text.strip()
    captcha_answer = debug_div.find_element(By.TAG_NAME, "code").text.strip()

    # Extract the rows using regex
    match = re.findall(r'Row\s+(\d+)\s*=\s*(\d+)', full_text)

    # Convert matches to a dict like {1: 0, 2: 1, 3: 1}
    row_values = {int(row): int(val) for row, val in match}

    # Assign to variables
    r1 = row_values.get(1)
    r2 = row_values.get(2)
    r3 = row_values.get(3)

    return r1, r2, r3, captcha_answer


# === UTILITY FUNCTIONS === #
def ensure_folder(folder):
    if os.path.exists(folder):
        for f in os.listdir(folder):
            path = os.path.join(folder, f)
            if os.path.isfile(path):
                os.remove(path)
            elif os.path.isdir(path):
                shutil.rmtree(path)
    else:
        os.makedirs(folder)

    # Recreate subfolders after clearing
    for sub in ['right', 'wrong']:
        os.makedirs(os.path.join(folder, sub), exist_ok=True)

# crop_center_captcha

def crop_center_captcha(driver):
    # Screenshot full page
    full = driver.get_screenshot_as_png()
    img = Image.open(BytesIO(full))

    # Manually tuned crop box (adjust if needed)
    top_left_x = 462
    top_left_y = 140
    width = 381
    height = 395

    return img.crop((top_left_x, top_left_y, top_left_x + width, top_left_y + height))

def get_row_wise_images(driver, row_number):
    # get full image of the captcha
    full = driver.get_screenshot_as_png()
    img = Image.open(BytesIO(full))

    #crop coordinates if row number = 1
    if row_number == 0:
        top_left_x = 462
        top_left_y = 140
        width = 381
        height = 395 // 3

    if row_number == 1:
        #crop coordinates if row number = 2
        top_left_x = 462
        top_left_y = 140 + 395 // 3
        width = 381
        height = 395 // 3

    if row_number == 2:
        #crop coordinates if row number = 3
        top_left_x = 462
        top_left_y = 140 + 2 * (395 // 3)
        width = 381
        height = 395 // 3
    return img.crop((top_left_x, top_left_y, top_left_x + width, top_left_y + height))


def save_captcha_image(captcha_img, index):
    if not os.path.exists(IMAGE_FOLDER):
        os.makedirs(IMAGE_FOLDER)
    file_path = os.path.join(IMAGE_FOLDER, f"captcha_{index}.png")
    captcha_img.save(file_path)
    return file_path

def solve_captcha():
    options = Options()
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)

    driver.get(HTML_FILE)
    time.sleep(1)

    r1, r2, r3, captcha_answer = extract_debug_info(driver)
    click_targets = {0: r1, 1: r2, 2: r3}

    next_buttons = {
        0: f"button[onclick=\"sendAction('next', '0')\"]",
        1: f"button[onclick=\"sendAction('next', '1')\"]",
        2: f"button[onclick=\"sendAction('next', '2')\"]"
    }
    reset_buttons = {
        0: f"button[onclick=\"sendAction('reset', '0')\"]",
        1: f"button[onclick=\"sendAction('reset', '1')\"]",
        2: f"button[onclick=\"sendAction('reset', '2')\"]"
    }

    for row in [0, 1, 2]:
        correct_clicks = click_targets[row]
        driver.find_element(By.CSS_SELECTOR, reset_buttons[row]).click()
        time.sleep(0.8)

        for i in range(6):
            # Capture the image of current row state
            img = get_row_wise_images(driver, row)

            # Decide folder: right or wrong
            folder = "right" if i == correct_clicks else "wrong"
            path = os.path.join(IMAGE_FOLDER, folder, f"row{row}_state{i}002.png")
            img.save(path)

            if i < 5:  # Don't click 'next' after last permutation
                driver.find_element(By.CSS_SELECTOR, next_buttons[row]).click()
                time.sleep(0.5)

    print("✅ All row permutations saved in right/wrong folders.")
    driver.quit()

if __name__ == "__main__":
    ensure_folder(IMAGE_FOLDER)
    solve_captcha()


