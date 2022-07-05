from __future__ import absolute_import

import dataclasses

import scrapy
from scrapy.item import Item, Field
from dataclasses import dataclass


@dataclass(repr=True)
class CarItem:
    # id : int
    used: bool
    make: str
    model: str
    year: str
    mileage: int
    body: str
    fuel: str
    capacity: int
    horsepower: int
    changeable_price: bool
    trade: bool
    dual_mass_flywheel: bool
    motor_emission_class: str
    drivetrain: str
    automatic_transmission: bool
    doors: str
    seats: str
    wheel_side_left: bool
    air_condition: str
    exterior_color: str
    interior_color: str
    interior_material: str
    registered_until: str
    is_registered: bool
    origin: str
    damaged: bool
    country: str
    leasing: bool
    loan: bool
    place: str
    link: str
    price: int

    def __init__(self, info):
        # self.id = info.get("Broj oglasa:", "?")
        self.used = info.get("Stanje", "Polovno vozilo") == "Polovno vozilo"
        self.make = info.get("Marka", "?")
        self.model = info.get("Model", "?")
        self.year = info.get("Godište", "?")
        self.mileage = int(info.get("Kilometraža", "?").replace(" km", "").replace(".", ""))
        self.body = info.get("Karoserija", "?")
        self.fuel = info.get("Gorivo", "?")
        self.capacity = int(info.get("Kubikaža", "?").replace(" cm3", ""))
        self.horsepower = int(info.get("Snaga motora", "?").split("/")[1].replace(" (kW", ""))
        self.changeable_price = info.get("Fiksna cena", "NE") != "DA"
        self.trade = info.get("Zamena", "DA") != "NE"
        self.dual_mass_flywheel = info.get("Plivajući zamajac", "?") == "Sa plivajućim zamajcem "
        self.motor_emission_class = info.get("Emisiona klasa motora", "?")
        self.drivetrain = info.get("Pogon", "?")
        self.automatic_transmission = info.get("Menjač", "?") == "Automatski / poluautomatski "
        self.doors = info.get("Broj vrata", "?")
        self.seats = info.get("Broj sedišta", "?")
        self.wheel_side_left = info.get("Strana volana", "Levi volan ") == "Levi volan "
        self.air_condition = info.get("Klima", "?")
        self.exterior_color = info.get("Boja", "?")
        self.interior_color = info.get("Boja enterijera", "?")
        self.interior_material = info.get("Materijal enterijera", "?")
        self.registered_until = info.get("Registrovan do", "?")
        self.is_registered = info.get("Registrovan do", "Nije registrovan ") != "Nije registrovan "
        self.origin = info.get("Poreklo vozila", "?")
        self.damaged = info.get("Oštećenje", "?") != "Nije oštećen "
        self.country = info.get("Zemlja uvoza", "?")
        self.loan = info.get("Lizing", "?") == "DA"
        self.leasing = info.get("Kredit", "?") == "DA"
        self.place = info.get("Grad", "?")
        self.link = info.get("Link", "?")
        self.price = info.get("Cena", -1)

    def __str__(self):
        key_value = dataclasses.asdict(self)

        values = [key_value.values()]

        return_str = ""
        for value in values:
            return_str += str(value)

        return_str = return_str.replace("[", "")
        return_str = return_str.replace("]", "")

        return return_str[11:]
