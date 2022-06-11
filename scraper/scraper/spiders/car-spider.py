from __future__ import absolute_import
import scrapy
from bs4 import BeautifulSoup
from ..items import CarItem


class CarSpider(scrapy.Spider):
    name = "cars"
    cars_counted = 0

    def start_requests(self):
        urls = [
            'https://www.polovniautomobili.com/auto-oglasi/pretraga?page=' + str(page) for page in range(10000)
        ]

        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse)
            if self.cars_counted >= 20000:
                break

    def parse(self, response):
        soup = BeautifulSoup(response.text, 'lxml')

        content = soup.find_all('article', {'class': 'ordinaryClassified'})

        detail_pages = [car_post.find_next('a')['href'] for car_post in content]

        for page in detail_pages:
            # print(page)
            yield response.follow(page, callback=self.parsePage)

    def parsePage(self, response):
        soup = BeautifulSoup(response.text, 'lxml')

        car_info = [x.text for x in soup.find_all('div', {'class': 'divider'})]

        car_info_cleaned = {}

        for info in car_info:
            key_value = info.strip("\n").split("\n")
            car_info_cleaned[key_value[0]] = key_value[1]

        car_object = CarItem(car_info_cleaned)
        self.cars_counted += 1

        return car_object