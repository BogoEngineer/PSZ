import plotly.express as px
import psycopg2
import plotly.graph_objects as go


class DataVisualization:
    def __init__(self, config):
        postgres_config = config['Postgres']
        self.postgres_connection = psycopg2.connect(database=postgres_config['Database'],
                                                    user=postgres_config['Username'],
                                                    password=postgres_config['Password'],
                                                    host=postgres_config['Host'],
                                                    port=postgres_config['Port'])
        self.postgres_cursor = self.postgres_connection.cursor()
        self.file_names = config['FileNames']['DataVisualization'].split(',')

    def top10_locations(self):
        sql_query = '''select place, count(*) from car group by place order by count(*) desc limit 10'''

        self.postgres_cursor.execute(sql_query)
        result = [r for r in self.postgres_cursor]

        self.visualize([r[0] for r in result], [r[1] for r in result], self.file_names[0], {'x': 'Location name', 'y': 'Number of cars'})

    def mileage_classes(self):
        sql_query = '''
            select 1 as seq, count(*)
            from car
            where mileage < 50000
            union 
            select 2 as seq, count(*)
            from car
            where mileage >= 50000 and mileage < 100000
            union 
            select 3 as seq, count(*)
            from car
            where mileage >= 100000 and mileage < 150000
            union
            select 4 as seq, count(*)
            from car
            where mileage >= 150000 and mileage < 200000
            union
            select 5 as seq, count(*)
            from car
            where mileage >= 200000 and mileage < 250000
            union
            select 6 as seq, count(*)
            from car
            where mileage >= 250000 and mileage < 300000
            union 
            select 7 as seq, count(*)
            from car
            where mileage >= 300000
            order by seq
            '''

        self.postgres_cursor.execute(sql_query)

        x = ['<50k', '50k-100k', '100k-150k', '150k-200k', '200k-250k', '250k-300k', '>300k']
        y = [r[1] for r in self.postgres_cursor]

        self.visualize(x, y, self.file_names[1], {'x': 'Distance travelled', 'y': 'Number of cars'})

    def year_classes(self):
        sql_query = '''
            select 1 as seq, count(*)
            from car
            where year < 1960
            union 
            select 2 as seq, count(*)
            from car
            where year > 1960 and year <= 1970
            union 
            select 3 as seq, count(*)
            from car
            where year > 1970 and year <= 1980
            union
            select 4 as seq, count(*)
            from car
            where year > 1980 and year <= 1990
            union
            select 5 as seq, count(*)
            from car
            where year > 1990 and year <= 2000
            union
            select 6 as seq, count(*)
            from car
            where year > 2000 and year <= 2005
            union 
            select 7 as seq, count(*)
            from car
            where year > 2005 and year <= 2010
            union 
            select 8 as seq, count(*)
            from car
            where year > 2010 and year <= 2015
            union 
            select 9 as seq, count(*)
            from car
            where year > 2015 and year <= 2020
            union 
            select 10 as seq, count(*)
            from car
            where year = 2021 or year = 2022
            order by seq
            '''

        self.postgres_cursor.execute(sql_query)

        x = ['1960', '1961-1970', '1971-1980', '1981-1990', '1991-2000', '2001-2005', '2006-2010', '2011-2015',
             '2016-2020', '2021-2022']
        y = [r[1] for r in self.postgres_cursor]

        self.visualize(x, y, self.file_names[2], {'x': 'Year range of manufacture', 'y': 'Number of cars'})

    def transmission_classes(self):
        sql_query = '''
            select 1 as seq, count(*)
            from car
            where automatic_transmission = true
            union
            select 2 as seq, count(*)
            from car
            where automatic_transmission = false
            order by seq
            '''

        self.postgres_cursor.execute(sql_query)

        x = ['automatic transmission', 'manual transmission']
        y = [r[1] for r in self.postgres_cursor]

        self.visualize(x, y, self.file_names[3], {'x': 'Type of transmission', 'y': 'Number of cars'})
        self.pie_chart(x, y, self.file_names[3])

    def price_classes(self):
        sql_query = '''
            select 1 as seq, count(*)
            from car
            where price < 2000 and leasing = false and loan = false
            union 
            select 2 as seq, count(*)
            from car
            where price >= 2000 and price < 5000 and leasing = false and loan = false
            union 
            select 3 as seq, count(*)
            from car
            where price >= 5000 and price < 10000 and leasing = false and loan = false
            union
            select 4 as seq, count(*)
            from car
            where price >= 10000 and price < 15000 and leasing = false and loan = false
            union
            select 5 as seq, count(*)
            from car
            where price >= 15000 and price < 20000 and leasing = false and loan = false
            union
            select 6 as seq, count(*)
            from car
            where price >= 20000 and price < 25000 and leasing = false and loan = false
            union
            select 7 as seq, count(*)
            from car
            where price >= 25000 and price < 30000 and leasing = false and loan = false
            union 
            select 8 as seq, count(*)
            from car
            where price >= 30000 and leasing = false and loan = false
            order by 1
            '''

        self.postgres_cursor.execute(sql_query)

        x = ['<2000', '2000-4999', '5000-9999', '10000-14999', '15000-19999', '20000-24999', '25000-29999', '>30000']
        y = [r[1] for r in self.postgres_cursor]

        self.visualize(x, y, self.file_names[4], {'x': 'Price range', 'y': 'Number of cars'})
        self.pie_chart(x, y, self.file_names[4])

    def visualize(self, x, y, file_name, label_names):
        fig = px.bar(x=x, y=y, labels=label_names)
        fig.write_html('{n}.html'.format(n=file_name), auto_open=True)

    def pie_chart(self, x, y, name):
        fig = go.Figure(data=[go.Pie(labels=x, values=y, textinfo='label+percent',
                                     insidetextorientation='radial'
                                     )])
        fig.write_html('{n}_pie.html'.format(n=name), auto_open=True)

    def close(self):
        self.postgres_cursor.close()
        self.postgres_connection.commit()
        self.postgres_connection.close()
