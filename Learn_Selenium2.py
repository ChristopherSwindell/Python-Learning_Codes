from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time

PATH = "C:\Program Files (x86)\chromedriver.exe"
driver = webdriver.Chrome(PATH)

driver.get("https://techwithtim.net")

##Navigate from home page to get started in the beginner python program, then back out
link = driver.find_element_by_link_text("Python Programming")
link.click()

try:
##Navigate to the tutorial
    element = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.LINK_TEXT, "Beginner Python Tutorials"))
    )
    ##Clears text in search bar
##    element.clear()

    ##click on element
    element.click()
    
    element = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.ID, "sow-button-19310003"))
    )
    element.click()
##No go back to the homepage
    driver.back()
    driver.back()
    driver.back()
##We can also go forward
    driver.forward()
    driver.forward()

except:
    driver.quit()
