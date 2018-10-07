import logging
import warnings
import time

from pyvirtualdisplay import Display
from selenium import webdriver

logging.basicConfig()
logger = logging.getLogger("LloydsCrawler")
logger.setLevel(logging.DEBUG)

warnings.filterwarnings("ignore")

display = Display(visible=0, size=(1920, 1080))
display.start()

profile = webdriver.FirefoxProfile()
profile.set_preference("general.useragent.override",
                               "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36")
profile.accept_untrusted_certs = True
driver = webdriver.Firefox(profile)

homepage_url = "http://q.10jqka.com.cn"
driver.get(homepage_url)
time.sleep(3)

url = "http://q.10jqka.com.cn/index/index/board/all/field/zdf/order/desc/page/2/ajax/1"
driver.get(url)
time.sleep(3)
driver.save_screenshot("takephoto1.png")
# WebDriverWait(driver, 30).until(EC.presence_of_element_located((By.CLASS_NAME, 'lli-searchform__input')))