from urllib.parse import urlparse, urlunparse
from itertools import product
import time

import re
import requests
from requests import ReadTimeout
from requests.exceptions import SSLError
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
from selenium.common.exceptions import WebDriverException, TimeoutException

from .ckeditor_paths_IoCs import paths_iocs
from .browser import Browser


def get_endpoint_flag_per_plugin(domain):
    endp_flags_per_plugin = []

    scheme, netloc, _, _, _, _ = urlparse(domain)
    for plugin in paths_iocs.keys():
        file_dict = paths_iocs[plugin]

        plugin_endpoints = []
        for file in file_dict.keys():
            endpoints = file_dict[file]['endpoints']
            entrypoints = file_dict[file]['entry_points']
            flags = file_dict[file].get('flags', '')
            plugin_endpoints.append((flags, [
                urlunparse((scheme, netloc, path_1 + '/' + path_2, '', '', '')) for (path_1, path_2) in list(product(entrypoints, endpoints))
            ]))
        endp_flags_per_plugin.append({
            'plugin': plugin,
            'endpoints': plugin_endpoints
        })

    return endp_flags_per_plugin


def get_path(uri):
    if not uri:
        return None

    _, _, path, _, _, _ = urlparse(uri)

    return [x.strip() for x in path.split('/') if len(x.strip()) > 0]


def run_chrome(url, screenshot_path, browser_config):
    benv = Browser(size=(1920, 1080))
    try:
        benv.start(browser_config)
        benv.driver.set_page_load_timeout(browser_config['timeout'])
        # print(url)
        benv.get(url)
        benv.driver.maximize_window()
        benv.save_screenshot(screenshot_path)
    except TimeoutException:
        print("This is taking too long!")
    except WebDriverException as e:
        print("Exception: " + e.msg)
    finally:
        benv.close()