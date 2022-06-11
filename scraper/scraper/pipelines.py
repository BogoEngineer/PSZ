import logging

import psycopg2


class PostgresPipeline:
    def __init__(self):
        self.postgres_cursor = None
        self.items = None
        self.postgres_connection = None

    def open_spider(self, spider):
        logging.info("Connecting to database: ")
        self.postgres_connection = psycopg2.connect(database="postgres", user="postgres", password="postgres",
                                                    host="127.0.0.1", port="5432")
        logging.info("Connected")
        self.postgres_cursor = self.postgres_connection.cursor()
        self.cleanDatabase()

        self.items = []

    def process_item(self, item, spider):
        self.items.append(item)
        if len(self.items) == 1000:
            logging.info("Saving 1000 records to database...")
            self.insertInto()
            self.items = []
        return item

    def close_spider(self, spider):
        # insert remaining records
        if self.items:
            logging.info("Saving remaining records to database...")
            self.insertInto()

        self.postgres_connection.commit()
        self.postgres_connection.close()

    def insertInto(self):
        sql_query = '''INSERT INTO car
                        VALUES'''
        for item in self.items:
            sql_query += str(item) + ','

        sql_query = sql_query[:-1]

        print(sql_query)

        self.postgres_cursor.execute(sql_query)

    def cleanDatabase(self):
        sql_query = '''DELETE FROM car'''
        self.postgres_cursor.execute(sql_query)
