from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time

PATH = "C:\Program Files (x86)\chromedriver.exe"
driver = webdriver.Chrome(PATH)

##Open webpage
driver.get("https://techwithtim.net")
print(driver.title)

##Use the search function on the webpage
search = driver.find_element_by_name("s")
search.send_keys("test")
search.send_keys(Keys.RETURN)

##We want to make sure that the page is finished loading before running the next script
try:
    main = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.ID, "main"))
    )
    articles = main.find_elements_by_tag_name("article")
    for article in articles:
        header = article.find_element_by_class_name("entry-summary")
        print(header.text)
finally:
    driver.quit()



##delays program by 5 seconds so that it doesn't end immediately so we can see the results
##time.sleep(5)



##Close tab
##driver.close()
##Close entire browser
##driver.quit()

