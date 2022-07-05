from __future__ import absolute_import
import scrapy
from bs4 import BeautifulSoup
from ..items import CarItem


class CarSpider(scrapy.Spider):
    name = "cars"
    baseUrl = 'https://www.polovniautomobili.com'

    def start_requests(self):
        urls = [
            '/auto-oglasi/pretraga?page=' + str(page) for page in range(10)
        ]

        for url in urls:
            yield scrapy.Request(url=self.baseUrl + url, callback=self.parse)

    def parse(self, response):
        soup = BeautifulSoup(response.text, 'lxml')

        content = soup.find_all('article', {'class': 'ordinaryClassified'})

        detail_pages = [car_post.find_next('a')['href'] for car_post in content]

        car_cities = [car_post.find_next('div', {'class': 'city'}).text for car_post in content]

        for page, car_city in zip(detail_pages, car_cities):
            yield response.follow(page, callback=self.parse_page, meta={'place': car_city[1:], 'link': self.baseUrl + page})

    def parse_page(self, response):
        soup = BeautifulSoup(response.text, 'lxml')

        car_info = [x.text for x in soup.find_all('div', {'class': 'divider'})]

        car_price = soup.find_all('span', {'class': 'priceClassified regularPriceColor'})[0].text[:-2].replace('.', '')

        car_info_cleaned = {'Grad': response.meta.get('place'), 'Link': response.meta.get('link'), 'Cena': int(car_price)}

        for info in car_info:
            key_value = info.strip("\n").split("\n")
            car_info_cleaned[key_value[0]] = key_value[1]

        car_object = CarItem(car_info_cleaned)

        return car_object
