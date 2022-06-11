from __future__ import absolute_import

import dataclasses

import scrapy
from scrapy.item import Item, Field
from dataclasses import dataclass


@dataclass(repr=True)
class CarItem:
    # id : int
    used : bool
    make : str
    model : str
    year : str
    mileage : str
    body : str
    fuel : str
    capacity : str
    horsepower : str
    changeable_price : bool
    trade : bool
    dual_mass_flywheel : bool
    motor_emission_class : str
    drivetrain : str
    automatic_transmission : bool
    doors : str
    seats : str
    wheel_side_left : bool
    air_condition : str
    exterior_color : str
    interior_color : str
    interior_material : str
    registered_until : str
    is_registered : bool
    origin : str
    damaged : bool
    country : str
    leasing : bool
    loan : bool

    def __init__(self, info):
        # self.id = info['Broj oglasa:'] if info.get('Broj oglasa') is not None else ""
        self.used = info['Stanje'] == "Polovno vozilo" if info.get('stanje') is not None else True
        self.make = info['Marka'] if info.get('Marka') is not None else ""
        self.model = info['Model'] if info.get('Model') is not None else ""
        self.year = info['Godište'] if info.get('Godište') is not None else ""
        self.mileage = info['Kilometraža'] if info.get('Kilometraža') is not None else ""
        self.body = info['Karoserija'] if info.get('Karoserija') is not None else ""
        self.fuel = info['Gorivo'] if info.get('Gorivo') is not None else ""
        self.capacity = info['Kubikaža'] if info.get('Kubikaža') is not None else ""
        self.horsepower = info['Snaga motora'] if info.get('Snaga motora') is not None else ""
        self.changeable_price = info['Fiksna cena'] != "DA" if info.get('Fiksna cena') is not None else True
        self.trade = info['Zamena'] != "NE" if info.get('Zamena') is not None else True
        self.dual_mass_flywheel = info['Plivajući zamajac'] == "Sa plivajućim zamajcem " if info.get('Plivajući zamajac') is not None else False
        self.motor_emission_class = info['Emisiona klasa motora'] if info.get(
            'Emisiona klasa motora') is not None else ""
        self.drivetrain = info['Pogon'] if info.get('Pogon') is not None else ""
        self.automatic_transmission = info['Menjač'] == "Automatski / poluautomatski " if info.get(
            'Menjač') is not None else False
        self.doors = info['Broj vrata'] if info.get('Broj vrata') is not None else ""
        self.seats = info['Broj sedišta'] if info.get('Broj sedišta') is not None else ""
        self.wheel_side_left = info['Strana volana'] == "Levi volan " if info.get('Strana volana') is not None else True
        self.air_condition = info['Klima'] if info.get('Klima') is not None else ""
        self.exterior_color = info['Boja'] if info.get('Boja') is not None else ""
        self.interior_color = info['Boja enterijera'] if info.get('Boja enterijera') is not None else ""
        self.interior_material = info['Materijal enterijera'] if info.get('Materijal enterijera') is not None else ""
        self.registered_until = info['Registrovan do'] if info.get('Registrovan do') is not None else ""
        self.is_registered = info['Registrovan do'] != "Nije registrovan " if info.get(
            'Registrovan do') is not None else False
        self.origin = info['Poreklo vozila'] if info.get('Poreklo vozila') is not None else ""
        self.damaged = info['Oštećenje'] != "Nije oštećen " if info.get('Oštećenje') is not None else True
        self.country = info['Zemlja uvoza'] if info.get('Zemlja uvoza') is not None else "?"
        self.loan = info['Lizing'] == "DA" if info.get('Lizing') is not None else False
        self.leasing = info['Kredit'] == "DA" if info.get('Kredit') is not None else False

    def __str__(self):
        key_value = dataclasses.asdict(self)

        values = [key_value.values()]

        return_str = ""
        for value in values:
            return_str += str(value)

        return_str = return_str.replace('[', '')
        return_str = return_str.replace(']', '')

        return return_str[11:]
