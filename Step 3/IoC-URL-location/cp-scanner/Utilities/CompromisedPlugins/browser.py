from selenium import webdriver
# from selenium.webdriver.common.proxy import Proxy, ProxyType
from pyvirtualdisplay import Display

from urllib.parse import urlparse
from PIL import Image, ImageFilter, ImageDraw
import io

import random
import re
import time
import os.path

from . import log
logger = log.getlogger("browser", log.DEBUG)


_CHROMEDRIVER_BINs = ["/path/to/chromedriver",
                      "/path/to/chromium-browser/chromedriver"]
CHROMEDRIVER_BIN = None

user_agent_list = [
    # Chrome
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/60.0.3112.113 Safari/537.36',
    'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/60.0.3112.90 Safari/537.36',
    'Mozilla/5.0 (Windows NT 5.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/60.0.3112.90 Safari/537.36',
    'Mozilla/5.0 (Windows NT 6.2; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/60.0.3112.90 Safari/537.36',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/44.0.2403.157 Safari/537.36',
    'Mozilla/5.0 (Windows NT 6.3; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/60.0.3112.113 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/57.0.2987.133 Safari/537.36',
    'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/57.0.2987.133 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/55.0.2883.87 Safari/537.36',
    'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/55.0.2883.87 Safari/537.36',
]


os.environ['DISPLAY'] = 'YOUR ENVIRONMENT VARIABLE'
for bin in _CHROMEDRIVER_BINs:
    if os.path.isfile(bin):
        CHROMEDRIVER_BIN = bin
        break

if not CHROMEDRIVER_BIN:
    raise Exception("chromedriver binary not found")


def _build_driver(user_agent, device_name=None, proxy=None, timeout=None, display=None, fullscreen=True):
    opts = webdriver.ChromeOptions()
    opts.add_experimental_option('prefs', {'intl.accept_languages': 'en,en_US'})
    opts.add_argument(user_agent)
    opts.add_argument('--headless')
    opts.add_argument('--no-sandbox')
    opts.add_argument('--ignore-certificate-errors')
    # opts.add_argument('ignore-certificate-errors-spki-list')
    logger.debug("Added load-extension")

    if proxy:
        opts.add_argument("proxy-server={}".format(proxy))
        logger.debug("Added proxy configuration")

    # We go full screen
    if fullscreen:
        opts.add_argument("start-fullscreen")
        logger.debug("Added start-fullscreen")

        # Remove the ugly yellow notification "Chrome is being controlled by automated test software"
    opts.add_argument("disable-infobars")

    # Set device name, or none
    if device_name:
        logger.debug("Enabling mobile emulation for {}".format(device_name))
        mobile_emulation = {"deviceName": device_name}
        opts.add_experimental_option("mobileEmulation", mobile_emulation)

    # Attach chrome to a specific display number
    if display:
        logger.debug("Attaching webdriver to a specific X display")
        opts.add_argument("display=:{}".format(display))

    logger.debug("Creating Chrome driver")
    driver = webdriver.Chrome(
        executable_path=CHROMEDRIVER_BIN, chrome_options=opts)

    if timeout:
        logger.debug("Setting page timeout")
        driver.set_page_load_timeout(timeout)

    return driver


class Browser:

    def __init__(self, size=(1920, 1080), device_name=None, port=None):
        self.size = size
        self.device_name = device_name

        self.vdisplay = None
        self.counter = None  # Request synchronized counter
        self.driver = None  # Web driver

    def start(self, browser_config):
        # virtual framebuffer
        # self.vdisplay = Display(visible=1, size=self.size)
        self.vdisplay = Display(visible=0, backend='xvfb', size=self.size)
        self.vdisplay.start()

        # create the driver and connect it to the proxy
        self.driver = _build_driver(**browser_config)

        rect = self.get_window_rect()
        self.real_size = (rect["width"], rect["height"])

    def get_window_rect(self):
        return self.driver.get_window_rect()

    def get(self, url):
        if self.driver is None:
            raise ValueError("Driver is not initialized: resources already released or not started.")

        self.driver.get(url)
        rect = self.get_window_rect()
        self.real_size = (rect["width"], rect["height"])

    def save_screenshot(self, image_filename):
        b = self.driver.get_screenshot_as_png()
        screenshot = Image.open(io.BytesIO(b))
        screenshot.save(image_filename)

    def close(self):
        if self.driver:
            self.driver.quit()

        if self.vdisplay:
            self.vdisplay.stop()

    def quit(self):
        self.close()



