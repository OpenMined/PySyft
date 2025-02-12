import os
import time

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys

chrome_driver_path = os.environ["CHROMEDRIVER_PATH"]
email = os.environ["NETFLIX_EMAIL"]
password = os.environ["NETFLIX_PASSWORD"]
profile = os.environ["NETFLIX_PROFILE"]
output_dir = os.environ["OUTPUT_DIR"]

print(f"🍿 Downloading Netflix Activity for: {email} Profile {profile}")

# Set up WebDriver (for Chrome)
chrome_options = Options()
prefs = {
    "download.default_directory": output_dir,
    "download.prompt_for_download": False,
}
chrome_options.add_experimental_option("prefs", prefs)
chrome_options.add_argument(
    "--headless"
)  # Run in headless mode, comment this if you want to see the browser window
chrome_service = Service(chrome_driver_path)  # Set the path to your ChromeDriver

driver = webdriver.Chrome(service=chrome_service, options=chrome_options)

# get login page
driver.get("https://www.netflix.com/login")


# Find the email and password input fields
email_input = driver.find_element(By.NAME, "userLoginId")
password_input = driver.find_element(By.NAME, "password")
# Enter email and password
email_input.send_keys(email)
password_input.send_keys(password)

# Submit the login form
print("Logging In")
password_input.send_keys(Keys.ENTER)

# Wait for the login to complete
time.sleep(3)

print("Switching Profiles")
# Navigate to Viewing Activity page
driver.get(f"https://www.netflix.com/SwitchProfile?tkn={profile}")

# Wait for the login to complete
time.sleep(3)

print("Getting Viewing Activity")
# Navigate to Viewing Activity page
driver.get("https://www.netflix.com/viewingactivity")

time.sleep(3)

print("Clicking Download all")
# Navigate to a page and download a file
element = driver.find_element(By.LINK_TEXT, "Download all").click()

print("Sleeping just in case")
time.sleep(10)

driver.quit()
