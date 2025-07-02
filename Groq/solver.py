# captcha_solver.py

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import os
from groq_utils import ask_llama_unscrambled
from PIL import Image
from io import BytesIO
import os, time, csv

HTML_FILE = "http://127.0.0.1:5000"
IMAGE_FOLDER = "captcha_images"

def ensure_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
    else:
        for file in os.listdir(folder):
            os.remove(os.path.join(folder, file))

def crop_row_image(driver, row):
    full_img = driver.get_screenshot_as_png()
    img = Image.open(BytesIO(full_img))

    top_left_x = 462
    top_left_y = 140 + row * (395 // 3)
    width = 381
    height = 395 // 3

    return img.crop((top_left_x, top_left_y, top_left_x + width, top_left_y + height))

def solve_captcha():
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service)
    driver.get(HTML_FILE)
    time.sleep(2)

    rows = [0, 1, 2]
    click_counts = {}

    next_btn = lambda row: f"button[onclick=\"sendAction('next', '{row}')\"]"
    reset_btn = lambda row: f"button[onclick=\"sendAction('reset', '{row}')\"]"

    for row in rows:
        driver.find_element(By.CSS_SELECTOR, reset_btn(row)).click()
        time.sleep(1)

        image_paths = []
        for i in range(6):
            img = crop_row_image(driver, row)
            fname = os.path.join(IMAGE_FOLDER, f"row{row}_state{i}.png")
            img.save(fname)
            image_paths.append(fname)

            if i < 5:
                driver.find_element(By.CSS_SELECTOR, next_btn(row)).click()
                time.sleep(0.8)

        unscrambled_index = ask_llama_unscrambled(image_paths)
        print(f"🧠 Row {row}: LLaMA says unscrambled index = {unscrambled_index}")
        click_counts[row] = unscrambled_index

    # Reset rows and apply correct clicks
    for row in rows:
        driver.find_element(By.CSS_SELECTOR, reset_btn(row)).click()
        time.sleep(0.8)
        for _ in range(click_counts[row]):
            driver.find_element(By.CSS_SELECTOR, next_btn(row)).click()
            time.sleep(0.6)

    print("✅ CAPTCHA solved with clicks:", click_counts)

    with open("click_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Row", "ClickCount"])
        for row, clicks in click_counts.items():
            writer.writerow([row, clicks])

    driver.quit()

if __name__ == "__main__":
    ensure_folder(IMAGE_FOLDER)
    solve_captcha()
