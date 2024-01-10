import asyncio
import configparser
import os
import time
from collections import namedtuple
from dataprep.connector import connect
from confluent_kafka import Producer
import json

# Environment variables
KAFKA_BROKER_URL = os.environ.get("KAFKA_BROKER_URL")
TOPIC_NAME = os.environ.get("TOPIC_NAME")
SLEEP_TIME = int(os.environ.get("SLEEP_TIME", 90))

# Configuration for OpenWeatherMap API
config = configparser.ConfigParser()
config.read('openweathermap_service.cfg')
api_credential = config['openweathermap_api_credential']
access_token = api_credential['access_token']

ApiInfo = namedtuple('ApiInfo', ['name', 'access_token'])
apiInfo = ApiInfo('openweathermap', access_token)

# Connect to the OpenWeatherMap API
sc = connect(apiInfo.name,
             _auth={'access_token': apiInfo.access_token},
             _concurrency=3)

async def get_weather(city):
    # Get the current weather details for the given city.
    df_weather = await sc.query("weather", q=city)
    return df_weather

def delivery_report(err, msg):
    if err is not None:
        print(f'Message delivery failed: {err}')
    else:
        print(f'Message delivered to {msg.topic()} [{msg.partition()}] at offset {msg.offset()}')

def run():
    locations = ['Ca Mau', 'Bac Lieu', 'Rach Gia', 'Soc Trang', 'Long Xuyen', 'Can Tho', 'Vinh Long', 'Cao Lanh', 'Ben Tre', 
                 'Tra Vinh', 'My Tho', 'Tan An', 'Ho Chi Minh City', 'Thu Dau Mot', 'Phu Khuong', 'Loc Ninh', 'Bien Hoa', 
                 'Vung Tau', 'Phan Thiet', 'Da Lat', 'Buon Ma Thuot', 'Phan Rang-Thap Cham', 'Nha Trang', 'Pleiku', 
                 'Kon Tum', 'Tuy Hoa', 'Quy Nhon', 'Turan', 'Quang Ngai', 'Hoi An', 'Hue', 'Quang Tri', 'Kwang Binh', 'Ha Tinh',
                 'Vinh', 'Thanh Hoa', 'Ninh Binh', 'Nam Dinh', 'Thai Binh', 'Phu Ly', 'Hung Yen', 'Hai Duong', 'Haiphong', 
                 'Ha Long', 'Lang Son', 'Cao Bang', 'Ha Giang', 'Lao Cai', 'Lai Chau', 'Son La', 'Dien Bien Phu', 'Ha Noi', 
                 'Hoa Binh', 'Viet Tri', 'Vinh Yen', 'Bac Giang', 'Bac Ninh', 'Yen Bai', 'Tuyen Quang', 'Thai Nguyen', 'Bac Kan']
    iterator = 0

    # Setup Producer with Confluent Kafka
    producer = Producer({
        'bootstrap.servers': KAFKA_BROKER_URL,
        'client.id': 'openweather_producer'
    })

    print("Setting up Weather producer at {}".format(KAFKA_BROKER_URL))

    while True:
        location = locations[iterator % len(locations)]
        current_weather = asyncio.run(get_weather(city=location))
        current_weather['location'] = location
        current_weather['report_time'] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

        # Convert DataFrame to JSON string
        current_weather = current_weather.to_json(orient="records")
        weather_json = current_weather[1:-1]
        
        # Print json
        print(weather_json)
      
        # Produce message
        producer.produce(TOPIC_NAME, value=weather_json, callback=delivery_report)
        print(f"Sent weather data for {location} to Kafka topic {TOPIC_NAME}")

        # Wait for any outstanding messages to be delivered and delivery report callbacks to be triggered.
        producer.poll(1)

        iterator += 1
        time.sleep(SLEEP_TIME)

    # Wait for all messages to be delivered
    producer.flush() 

if __name__ == "__main__":
    run()