from selenium import webdriver
from selenium.webdriver.edge.service import Service
from selenium.webdriver.common.by import By
import time

# Start Edge browser
s = Service("E:/Downloads/msedgedriver.exe")
driver = webdriver.Edge(service=s)

# Open Smartprix mobiles page
driver.get("https://www.smartprix.com/mobiles")

old_height=driver.execute_script("return document.body.scrollHeight")

while True:
    driver.find_element(by=By.XPATH,value='//*[@id="app"]/main/div[1]/div[2]/div[3]').click()
    time.sleep(2)
    new_height=driver.execute_script("return document.body.scrollHeight")
    print(new_height)
    print(old_height)
    if new_height==old_height:
        break
    old_height=new_height

html=driver.page_source

with open("smartprix.html","w",encoding="utf-8") as f:
    f.write(html)