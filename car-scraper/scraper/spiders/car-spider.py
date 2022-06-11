from __future__ import absolute_import
import scrapy
from bs4 import BeautifulSoup


class CarSpider(scrapy.Spider):
    name = "cars"

    def start_requests(self):
        urls = [
            # 'https://www.polovniautomobili.com/',
            'https://www.polovniautomobili.com/auto-oglasi/pretraga?page=1'
        ]

        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response):
        soup = BeautifulSoup(response.text, 'lxml')

        content = soup.find_all('article', {'class': 'ordinaryClassified'})

        detail_pages = [car_post.find_next('a')['href'] for car_post in content]

        for page in detail_pages:
            print(page)
            yield response.follow(page, callback=self.parsePage)

    def parsePage(self, response):
        soup = BeautifulSoup(response.text, 'lxml')

        car_info = [x.text for x in soup.find_all('div', {'class': 'divider'})]

        car_info_cleaned = {}

        for info in car_info:
            key_value = info.strip("\n").split("\n")
            car_info_cleaned[key_value[0]] = key_value[1]

        car_object = CarItem(car_info_cleaned)

        print('----------------------------------???')
        # for child in basic_info.children:
        #     print(child)
        print(car_info_cleaned)
        print(len(car_info_cleaned))
        # print(y for y in (x.children for x in car_info))
        print('----------------------------------')
