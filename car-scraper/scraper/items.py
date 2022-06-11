from __future__ import absolute_import
import scrapy
from scrapy.item import Item, Field


class CarItem(scrapy.Item):
    id = Field()
    used = Field()
    make = Field()
    model = Field()
    year = Field()
    mileage = Field()
    body = Field()
    fuel = Field()
    capacity = Field()
    horsepower = Field()
    changeable_price = Field()
    trade = Field()
    dual_mass_flywheel = Field()
    motor_emission_class = Field()
    drivetrain = Field()
    automatic_transmission = Field()
    doors = Field()
    seats = Field()
    wheel_side_left = Field()
    air_condition = Field()
    exterior_color = Field()
    interior_color = Field()
    interior_material = Field()
    registered_until = Field()
    origin = Field()
    damaged = Field()
    country = Field()
    leasing = Field()
    loan = Field()

    def __init__(self, info):
        self.id = info['Broj oglasa:'] if info['Broj oglasa'] else ""
        self.used = info['Polovno vozilo'] if info['Polovno vozilo'] else ""
        self.make = info['Marka'] if info['Marka'] else ""
        self.model = info['Model'] if info['Model'] else ""
        self.year = info['Godište'] if info['Godište'] else ""
        self.mileage = info['Kilometraža'] if info['Kilometraža'] else ""
        self.body = info['Karoserija'] if info['Karoserija'] else ""
        self.fuel = info['Gorivo'] if info['Gorivo'] else ""
        self.capacity = info['Kubikaža'] if info['Kubikaža'] else ""
        self.horsepower = info['Snaga motora'] if info['Snaga motora'] else ""
        self.changeable_price = info['Fiksna cena'] != "DA" if info['Fiksna cena'] else True
        self.trade = info['Zamena'] != "NE" if info['Zamena'] else True
        self.dual_mass_flywheel = info['Plivajući zamajac'] if info['Plivajući zamajac'] else ""
        self.motor_emission_class = info['Emisiona klasa motora'] if info['Emisiona klasa motora'] else ""
        self.drivetrain = info['Pogon'] if info['Pogon'] else ""
        self.automatic_transmission = info['Menjač'] == "Automatski / poluautomatski " if info['Menjač'] else False
        self.doors = info['Broj vrata'] if info['Broj vrata'] else ""
        self.seats = info['Broj sedišta'] if info['Broj sedišta'] else ""
        self.wheel_side_left = info['Strana volana'] == "Levi volan " if info['Strana volana'] else True
        self.air_condition = info['Klima'] if info['Klima'] else ""
        self.exterior_color = info['Boja'] if info['Boja'] else ""
        self.interior_color = info['Boja enterijera'] if info['Boja enterijera'] else ""
        self.interior_material = info['Materijal enterijera'] if info['Materijal enterijera'] else ""
        self.registered_until = info['Registrovan do'] if info['Registrovan do'] else ""
        self.is_registered = info['Registrovan do'] != "Nije registrovan " if info['Registrovan do'] else False
        self.origin = info['Poreklo vozila'] if info['Poreklo vozila'] else ""
        self.damaged = info['Oštećenje'] != "Nije oštećen " if info['Oštećenje'] else True
        self.country = info['Zemlja uvoza'] if info['Zemlja uvoza'] else "Srbija"
        self.loan = info['Lizing'] == "DA" if info['Lizing'] else False
        self.leasing = info['Kredit'] == "DA" if info['Kredit'] else False

